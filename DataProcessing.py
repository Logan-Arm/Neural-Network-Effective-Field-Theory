import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import networkx as nx
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level = logging.INFO)
#Adjust above to change the logger output level


def fixed_sign_evecs(evecs, reference_evecs, e_amount = None):
    """A helper function used to define a base orientation of the eigenvectors.  As np.linalg computes eigenvectors correctly up to a
    -1 factor, when we take derivatives of any sort of eigenvector term (such as the residuals), we introduce a lot of volatility unless we 
    fix the eigenvectors to a specific orientation.  
    
    If a reference eigenvector (the base eigenvector) is given, the eigenvectors sign are calculated by dotting with the reference eigenvector.
    If no reference eigenvector is inputted, the sign of the eigenvector is then dertermined by the sign of the largest component within the eigenvector.

    It is recommended that this function be used to define the reference eigenvector once (for each eigenvector/replica), and then continue using that
    calculated reference eigenvector going forward. 
    """
    if e_amount is None:
        e_amount = evecs.shape[1]
    for k in range(e_amount):
        vec = evecs[:,k]
        if reference_evecs is not None:
            ref_vec = reference_evecs[:,k]
            sign = np.sign(np.dot(vec, ref_vec))
        else:
            """If no reference vectors are assigned, take the sign convention for each individual vector to be the sign of the largest singular component"""
            sign = np.sign(vec[np.argmax(np.abs(vec))])
        if sign ==0:
            sign =1
        evecs[:,k] *=sign
    return evecs


class DataHandler():
    def __init__(self,trial,bootstraps, e_amount, p_target, output_dir):
        self.trial = trial
        self.bootstraps = bootstraps
        self.e_amount = e_amount
        """Need amount of relevant eigenvalues to calculate"""
        self.numlayer = self.trial.numlayer
        """Number of layers is needed for organisation"""
        self.p_target = p_target
        """Need target phi to calculate when phi crosses"""
        self.ensemble = self.trial.ensemble
        """Ensemble number relevant for any error on mean"""
        self.output_dir = Path(output_dir)

        self.y_train = self.trial.y_train.squeeze().numpy()
        self.norm_const = self.trial.norm_const

        self.phi_target_time_layer = []
        self.phi_target_time_neuron = np.zeros((self.numlayer-1,self.trial.width))

        self.param_dict_length = self.trial.parameter_recorder.numlayer

        self.compute_trial_averages()
        logger.debug("DataHandler Initialised")

    def compute_trial_averages(self):
        
        """A function which simply takes the averages of some of the key quantities recorded during the trial.
        These values include the NTK, plateau start and end, and performances"""

        logger.info("Computing trial averages")

        self.ens_avg_NTK = np.mean(self.trial.ntk_recorder.ensemble_NTK, axis =0) #axis along the ensemble dim, remaining dim should be [time, 16,16]
        """NTK uncertainty on it's own is not a particularly helpful metric on its own, 
        any of the further calculations derived from NTK (i.e., eigenvalues) have uncertainties better modeled using bootstrapping
        """
        self.ens_avg_loss = np.mean(self.trial.loss_recorder.ensemble_loss, axis = 0)
        self.ens_loss_uncert = np.std(self.trial.loss_recorder.ensemble_loss, axis = 0)/np.sqrt(self.ensemble)

        self.ens_avg_perf = np.mean(self.trial.performance_recorder.performances, axis = 0)
        self.ens_perf_uncert =np.std(self.trial.performance_recorder.performances, axis =0)/np.sqrt(self.ensemble) 

    def compute_eigvals_zero(self):
        """A function to compute the ensemble average eigenvalues and eigenvectors at initilaisation, with bootstrapped uncertainty"""

        logger.info("Starting compute_eigvals_zero")
        NTK_stacked = np.stack(self.trial.ntk_recorder.initial_NTK, axis =0)        

        
        bootstrapped_eigvals = np.zeros((self.bootstraps, self.e_amount)) 
        bootstrapped_eigvecs = np.zeros((self.bootstraps, len(self.trial.x_train), self.e_amount))

        reference_evecs =None #Same logic will be applied here as in the ntk_recorder
        """Bootstrapping loop"""
        for b in range(self.bootstraps):
            #Recall NTK_stacked zero axis is our replica axis
            boots_indices = np.random.choice(NTK_stacked.shape[0], size = NTK_stacked.shape[0], replace = True)
            #Randomly resampling (with replacement) which replicas to include
            boot_mean_NTK = np.mean(NTK_stacked[boots_indices], axis = 0) 
            #Taking mean of this resampled
            boot_mean_NTK = (boot_mean_NTK + boot_mean_NTK.T)/2 #Making explicitly symmetric, recall that the NTK is supposed to be symmetric
            boot_eigvals, boot_eigvecs = np.linalg.eigh(boot_mean_NTK)
            if b ==0:
                logger.debug("Bootstrapped eigenvectors have dimensionality %s",boot_eigvecs.ndim)
            boot_sorted_indices = np.argsort(boot_eigvals)[-self.e_amount:][::-1] #Grabs the last self.e points in descending order
            boot_eigvals = boot_eigvals[boot_sorted_indices]
            boot_eigvecs = boot_eigvecs[:,boot_sorted_indices] #Need slice to actually take the vector
            if b ==0:
                logger.debug("the first bootstrapped eigenvector (boot_eigvecs) has shape y %s, x %s", boot_eigvecs[0,:].shape,boot_eigvecs[:,0].shape )
            bootstrapped_eigvals[b] = boot_eigvals

            boot_eigvecs_signed = fixed_sign_evecs(boot_eigvecs, reference_evecs,e_amount= self.e_amount)
            if reference_evecs is None:
                reference_evecs = boot_eigvecs_signed

            bootstrapped_eigvecs[b] = boot_eigvecs_signed

            if b%10 ==0:
                logger.debug("Bootstrap %d done", b)

        
        
        self.mean_eigenvals = np.mean(bootstrapped_eigvals, axis = 0)
        self.mean_eigenvals_uncert = np.std(bootstrapped_eigvals, axis =0)
    
        # print(f"The eigenvalues of interest are {self.mean_eigenvals}")
        logger.info("The eigenvalues of interest are %s", self.mean_eigenvals)

        self.mean_eigenvecs = np.mean(bootstrapped_eigvecs, axis =0)
        logger.debug("The mean bootstrapped first eigenvector has shape %s", self.mean_eigenvecs[:,0].shape)
        logger.debug("The first eigenvector is %s",self.mean_eigenvecs[:,0])
        logger.debug("The second eigenvector is %s",self.mean_eigenvecs[:,1])

    def compute_eigen_quant(self):
        """Calculates the ensemble wide eigenquantities over time.  Does this by taking the mean over replicas
        for the ensemble NTK from the ntk_recorder in the trial_recorders_model code.  For each instance of a recorded
        NTK, (as defined by the alignmentint), the eigenvalues and eigenvectors are recorded.  Using this, the 
        eigenvecctor/values are recorded, as well as the NTK alignment.  No bootstrapped uncertainty has been implemented yet, 
        although this could be accomplished by bootrapping the NTK at each alignmentint"""
        logger.info("Starting compute_eigen_quant")

        ensemble_evals = []
        ensemble_evecs = []
        ensemble_NTK_align = []

        reference_evecs = None

        for i in range(self.ens_avg_NTK.shape[0]):
            inst_NTK = self.ens_avg_NTK[i,:,:]
            ntk_norm = np.linalg.norm(inst_NTK,'fro')
            eigvals,eigvecs = np.linalg.eigh(inst_NTK)
            idx = np.argsort(eigvals)[-self.e_amount:][::-1]

            evals = eigvals[idx]
            evecs_non_aligned = eigvecs[:,idx]
            if reference_evecs is not None:
                evecs_aligned = fixed_sign_evecs(evecs_non_aligned, reference_evecs)
            else:
                evecs_aligned = fixed_sign_evecs(evecs_non_aligned,None)
                reference_evecs = evecs_aligned
                logger.debug("Reference Evecs Assigned")

            alignment_item = np.dot(self.y_train, inst_NTK)
            alignment_append = np.dot(alignment_item,self.y_train.T)/(ntk_norm *self.norm_const)
            ensemble_evals.append(evals)
            ensemble_evecs.append(evecs_aligned)
            ensemble_NTK_align.append(alignment_append)

            logger.debug("Computed Eigen Quantities for Replica Number %d",i)

        self.ensemble_evals = np.array(ensemble_evals)
        self.ensemble_evecs = np.array(ensemble_evecs)
        self.ensemble_NTK_align = np.array(ensemble_NTK_align)

    def compute_ensemble_evec_rotation(self):
        """Calculates the ensemble average eigenvector rotation from the initial eigenvector direction over time, does so using 
         standard dot product formula """

        logger.info("Starting compute_ensemble_evec_rotations")

        initial_evecs = self.ensemble_evecs[0,:,:]
        dot_products = np.einsum('evi,vi->ei', self.ensemble_evecs, initial_evecs) #contract over the vector index (i,e., axis 1 for ensemble, 0 for initial)
        logger.debug("Dimensionality of the dot products are %s", dot_products.shape)

        initial_evecs_norms = np.linalg.norm(initial_evecs, axis =0)

        logger.debug("Dimensionality of the initial_evecs_norms is %s",initial_evecs_norms.shape)#expect it to be (self.e_amount,)
        eigenvector_norms = np.linalg.norm(self.ensemble_evecs, axis =1) #dimensionality [epoch, norm, index]

        logger.debug("Dimensionality of eigenvector norms is %s",eigenvector_norms.shape)
        logger.debug("Dimensionality of eigenvector_norms[:,] * initial_evec_norms is %s",(eigenvector_norms[:,]* initial_evecs_norms).shape)

        cos_angles = np.abs(dot_products)/(eigenvector_norms[:,] * initial_evecs_norms)
        #Of dimensionality [epochs, e_amount], -->think right
        self.evec_rotation_mean = np.arccos(np.clip(cos_angles,-1,1))

    def rotation_help(self, base, vector, base_norm=None ):
        """A helper function to calculate the rotation of an inputted vector from an inputted base using standard dot product identity """
        if base_norm is None:
            base_norm = np.linalg.norm(base)

        vector_norm = np.linalg.norm(vector)
        dot_product = np.dot(vector, base)

        cos_angle = np.abs(dot_product)/(vector_norm*base_norm)

        angle = np.arccos(np.clip(cos_angle,-1,1))
        return angle

    def replica_sort(self):
        """A helper function to put replicas in ascending order in terms of plateau end time, i.e.,
        replica 1 has earliest plateau end time, replica 2 has 2nd earliest plateau end time, so on and so forth"""
        plateau_end_times = self.trial.plateau_recorder.plateau_bounds[:,1]
        sorted_indices = np.argsort(plateau_end_times)
        return sorted_indices

    def replica_TOI_helper(self,num_times=3,record_evals =True, num_evals =3, sort = True):
        """Helper function to return the required indices of NTK's of each replica at specified times
        
        
        NTK theory suggests that the relevant timescales to the learning process are as follows:
        [0,t1], where t1 = 1/lambda1(0),
        [t1, t2], where t2 = 1/lambda2(t1),
        [t2,t3], where t3 = 1/lambda3(t2),
        etc

        To confirm that this behavior is observed experimentally, these times must be calculated on a replica by replica basis; as it has been observed that
        plateau length is not universal, and is highly tied to initilisation, any sort of ensemble average would be inaccurate (similar to the plateau end identification)  
        
        
        """
        x_len = len(self.trial.x_train.squeeze())

        self.replica_TOI_indices = np.empty((self.ensemble,num_times)) #dimensionality [replicas, num_times]
        self.replica_TOI_times = np.full((self.ensemble,num_times),np.nan)
        if record_evals is True:
            self.replica_evals_TOI = np.full((self.ensemble, num_times, num_evals), np.nan) #[replicas, TOIs, num_evals]

        if sort:
            indices = self.replica_sort()
        else:
            indices = np.arange(self.ensemble)

        self.replica_sort_indices_replicaTOI = indices
        #array to store all the indices corresponding to NTK's at t1, t2
        for rank in range(self.ensemble):
            replica_idx = indices[rank]
            time_index = 0
            for j in range(num_times):
                if time_index is None:
                    self.replica_TOI_indices[rank,j] = np.nan
                    print("NONCONVERGENCE")
                    logger.info("Nonconvergence met for replica %d, time %d", rank, j)
                    continue
                NTK = self.trial.ntk_recorder.ensemble_NTK[replica_idx, time_index,:]
                eval_term = j+1
                if record_evals:
                    evals = np.linalg.eigvalsh(NTK)

                    eval = evals[-eval_term]
                    #eval corresponds to the eigenvalue designating the time index
                    evals = evals[::-1][:num_evals]
                    self.replica_evals_TOI[rank,j,:] = evals
                    
                else:
                    eval = np.linalg.eigvalsh(NTK)[-eval_term]
                time = 1/eval
                self.replica_TOI_times[rank,j] = time
                #Storing the raw time

                time_index = self.convert_times(time, training_time=True)
                self.replica_TOI_indices[rank,j] = time_index #Want to store t1, t2, t3 etc
                #Storing the index (to nearest alignment int) corresponding to this time

                if j==(num_times-1):
                    print(f"t3 raw time is {int(time)}")
                    print(f"t3 indexed time is {time_index}")


        print(self.replica_TOI_indices)
        logger.info("Completing replica_TOI_helper function")

        """Current configuration stores times of interest up to num_times, and inserts an nan in any case where a time of interest is larger than the maximum
        possible training time.  Once a replica is considered to have not converged (i.e., a time of interest being larger than the maximum possible time
        ), no further checks are done to see if later possible times of interest would be within the bounds.  I believe this convention should hold in all cases"""
                    
    def compute_TOI_evec_alignment(self, num_evals =3, sort = True):
        """A function meant to calculate the dot product between NTK eigenvectors of replicas at each replica's corresponding t1,t2 etc
        This function requires individual_replica_TOI be called prior, as it utilizes the matrices created in that function
        """
        x_len = len(self.trial.x_train.squeeze())
        all_NTKs = self.trial.ntk_recorder.ensemble_NTK
        number_TOI = self.replica_TOI_indices.shape[1]
        logger.info("compute_TOI_evec_alignment number_TOI is %d", number_TOI)

        # evec_1_overlap = np.empty((self.ensemble,self.ensemble,))
        logger.info("Compute_TOI_evec_alignment calculating individual eigenvectors")
        abs_replica_evecs = np.full((self.ensemble, number_TOI,x_len,num_evals),np.nan)

        """Abs_replica_evecs has following dimensionality : [ensemble, number of TOI, length of eigenvector, number of eigenvectors of interest]
        We use np.full with np.nan instead of np.empty to avoid the following case:
        Where a replica does not have a meaningul TOI, it is passed in the calculation, leaving whatever values initially filling the array as the vector array
        When we later use np.einsum this will create fake eigenvector alignment data
        Using np.zeros seems like a reasonable fit, but there is a possibility of convergent replicas having a 0 alignment at some given time, this then 
        corrupts the reading of that data

        np.nan gets around this as we can set the "bad color map" for seaborn heatmaps explicitly, allowing an easy way of filtering out nonconvergent replicas
        """
        if sort:
            indices = self.replica_sort()
        else:
            indices = np.arange(self.ensemble)

        self.replica_sort_indices_replicaTOI_evecALign = indices

        for rank in range(self.ensemble):
            NTK_idx = indices[rank]
            for j in range(number_TOI):
                if np.isnan(self.replica_TOI_indices[rank,j]):
                    continue
                    #Recall that self.replica_TOI_indices stores nans wherever a time of interest is beyond the maximum training time
                inst_index = int(self.replica_TOI_indices[rank,j])
                inst_ntk = all_NTKs[NTK_idx,inst_index,:]

                discard, evecs = np.linalg.eigh(inst_ntk)
                descending_indices = np.argsort(discard)[::-1][:num_evals]
                #argsort puts in ascending order, [::-1] revereses the order, then we take up to the number of evals

                evecs_sorted = evecs[:,descending_indices]
                print(f"Dimensionality of evecs_sorted is {evecs_sorted.shape}")
                abs_replica_evecs[rank,j,:] = np.abs(evecs_sorted)
                logger.debug("Computed eigenvectors for replica %d, time %d",NTK_idx ,j)

        overlap_all = np.einsum('ajxk, bjxk -> kjab', abs_replica_evecs, abs_replica_evecs)
        """a and b are separate replica indices, they are preserved to maintain heatmap information.  K corresponds to the eigenvector index,
        this has been set to the dict identifier key in the definition of evec_overlap, hence why it has been brought to the front, j is the time index"""

        self.evec_overlap_TOI = {k: overlap_all[k] for k in range(num_evals)}

    def compute_ensAVG_evec_alignment_TOI(self,num_evals =3, sort = True):
        """Function to compute the eigenvector alignment on a replica by replica basis at times of interest defined by the 
        ENSEMBLE NTK's times of interest, not replica by replica NTK's"""
        x_len = len(self.trial.x_train.squeeze())
        all_NTKs = self.trial.ntk_recorder.ensemble_NTK
        number_TOI = len(self.ensemble_TOIs)
        logger.info("compute_ensAVG_evec_alignment_TOI is %s", number_TOI)

        # evec_1_overlap = np.empty((self.ensemble,self.ensemble,))
        logger.info("compute_ensAVG_evec_alignment_TOI calculating individual eigenvectors")
        abs_replica_evecs = np.full((self.ensemble, number_TOI,x_len,num_evals),np.nan)

        if sort:
            indices = self.replica_sort()
        else:
            indices = np.arange(self.ensemble)        

        self.replica_sort_indices_TOI_ensemble_align = indices
        for j in range(number_TOI):
            if self.ensemble_TOIs[j] is None:
                logger.info("Ensemble time of interest %s is nonetype", self.ensemble_TOIs[j])
                continue
            time_index = int(self.ensemble_TOIs[j])
            for rank in range(self.ensemble):
                NTK_idx = indices[rank]
                inst_ntk = all_NTKs[NTK_idx,time_index,:]

                discard, evecs = np.linalg.eigh(inst_ntk)
                descending_indices = np.argsort(discard)[::-1][:num_evals]
                #argsort puts in ascending order, [::-1] revereses the order, then we take up to the number of evals

                evecs_sorted = evecs[:,descending_indices]
                abs_replica_evecs[rank,j,:] = np.abs(evecs_sorted)

        overlap_all = np.einsum('ajxk, bjxk -> kjab', abs_replica_evecs, abs_replica_evecs)
        """a and b are separate replica indices, they are preserved to maintain heatmap information.  K corresponds to the eigenvector index,
        this has been set to the dict identifier key in the definition of evec_overlap, hence why it has been brought to the front, j is the time index"""

        self.ensembleAVG_evec_overlap_TOI = {k: overlap_all[k] for k in range(num_evals)}

    def compute_ensemble_timesOI(self, num_times = 3):
        """Computes the times of interest on an ensemble average NTK basis
        """

        logger.info("Computing ensemble TOIs")
        self.ensemble_TOIs = []
        index_num = 0
        for i in range(num_times):
            if index_num is None:
                continue
            eval_term = i+1
            inst_NTK = self.ens_avg_NTK[index_num,:]

            eval_of_interest = np.linalg.eigvalsh(inst_NTK)[-eval_term]

            time = 1/eval_of_interest

            converted_index = self.convert_times(time, training_time=True)
            self.ensemble_TOIs.append(converted_index)
            index_num = converted_index

    def convergence_filter_helper(self):
        """Helper function to create a mask of replicas which did or did not converge"""
        plat_end_times = self.trial.plateau_recorder.plateau_bounds[:,1]
        mask = ~np.isnan(plat_end_times)
        return mask

    def compute_ens_eval3_t2(self, maxtimefilter = False):
        """Computes the ensemble average of eval3(t2), in both a filtered case and a unfiltered case.  
        The filtered case corresponds to the ensemble average only of replicas which did finish/leave their plateau,
        whereas the unfiltered case is simply the average over all the replicas
        
        
        """
        logger.info("Computing ens_eval3_t2")
        eval3_t2 = self.replica_evals_TOI[:,1,2]
        self.eval_3_mean = np.mean(eval3_t2)

        mask = self.convergence_filter_helper()
        self.filtered_eval3 = eval3_t2[mask]
        self.filtered_eval3_mean = np.mean(self.filtered_eval3)
        

        if maxtimefilter is True:
            t3_temp = self.replica_TOI_times[:,2]
            t3_temp_masked = t3_temp[mask] #have to apply the mask beforehand, and then filter out anything beyond the max training time
            t3_temp_masked[t3_temp_masked > (self.trial.lr *self.trial.epochs)] = np.nan

            self.filtered_t3_mean = np.nanmean(t3_temp_masked) #takes the nan mean of the t3 values filtered first by plateau convergence, then by max training time
            self.t3_mean = np.nanmean(self.times_OI[:,2])

        else:
            t3 = self.replica_TOI_times[:,2] 
            print(f"t3 is {t3}")
            print(f"t3 masked is {t3[mask]}")
            self.filtered_t3_mean = np.nanmean(t3[mask])
            self.t3_mean = np.nanmean(t3)

    def convert_times(self, time, training_time = False):
        """Function meant to convert inputted epochs/training times to the closest appropriate measured epoch under the alignmentint criteria
         if training_time set to True, it will be considered that the inputted time is in terms of training time, and thus modified by the lr (case for timesOI)
           
           Take epoch, divide by ten, round to nearest alignment int number, and return that index of the recorded epochs
           
            Note: this is an unstable function, and it is not recommended that this is used beyond searching for t2. There is a built in assertion to ensure that we 
            never search for values which are not recorded, but this is often going to be the case when searching for index values beyond t2, as evidenced by the instability
            in eval 3. 

           """
        if training_time is True:
            time=time/self.trial.lr # convert it back into epochs


        #Debug check to make sure that we are never searching for values that we cannot search for
        max_recorded = self.trial.ntk_recorder.recorded_epochs[-1]
        if time>max_recorded:
            return None


        time_aligned = time/self.trial.ntk_recorder.alignmentint #divide the epoch by the alignment int
        number_recordings = np.round(time_aligned).astype(int) #round epoch/alignment int to nearest whole number
        time_rounded = int(number_recordings*self.trial.ntk_recorder.alignmentint) #Conver this back into alignmentint epochs

        assert time_rounded <= self.trial.ntk_recorder.recorded_epochs[-1], "Time_rounded is beyond the maximum epoch recorded"
        
        #Not sure the .index works on numpy arrays, may need to to_list it
        index = np.searchsorted(self.trial.ntk_recorder.recorded_epochs,time_rounded)
        assert index <= len(self.trial.ntk_recorder.recorded_epochs), "Index outside range of array"
        # print(f"Time rounded is {time_rounded} and the indexed time is {self.trial.ntk_recorder.recorded_epochs[index]}")
        assert time_rounded == self.trial.ntk_recorder.recorded_epochs[index], "Time rounded does not equal the indexed value"
        return index

    def save_data(self):
        params ={
            'InputSize': [self.trial.input],
            'OutputSize': [self.trial.output],
            'HiddenLayerWidth': [self.trial.width],
            'HiddenLayerDepth': [self.trial.depth],
            'LearningRate': [self.trial.lr],
            'Epochs': [self.trial.epochs],
            'STD': [self.trial.std],
            'EnsembleNum': [self.ensemble],
            'Bootstraps': [self.bootstraps],
            'AlignmentInterval': [self.trial.ntk_recorder.alignmentint],
            'TargetPhi': [self.p_target]
        }
        df = pd.DataFrame(params)
        # df.to_csv(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\Params', index = False)
        df.to_csv(self.output_dir/"Params", index = False)
    
    def compute_neurons_plateau_info(self):
        """
        Function to determine the change in number of "active" neurons from plateau start to plateau end
        Recall that trial.phi has following dimensionality
        [layer][replica][epoch,neuron]

        Right now, no special care is being taken to ensure that the epochs from plateau_bounds will line up with self.phi. 
        At the time of writing, I am only focused on recording all data, and so this will not be a problem, but if there is ever a situation where hooks
        are not appended at the start, this WILL be wrong.
        """
        self.active_neur_start = np.zeros((self.ensemble,self.numlayer-1 )) #1 array per layer (excluding final) per number of replicas
        self.change_active_neur = np.zeros((self.ensemble,self.numlayer-1 ))
        self.plateau_lengths = []
        print(f'on compute_neurons_plateau')
        logger.info("Computing neurons_plateau")
        logger.debug("Trial phi array keys are %s", self.trial.phi.keys())
        for j in range(self.ensemble):
            epochs_OI = self.trial.plateau_recorder.plateau_bounds[j]
            if np.any(np.isnan(epochs_OI)):
                continue
            epoch_start = int(epochs_OI[0])
            epoch_end = int(epochs_OI[1])
            self.plateau_lengths.append(epoch_end-epoch_start)


            for k in range(self.numlayer-1):
                activity= self.trial.phi[k]
                activity = activity[j]
                activity_start = np.sum(activity[epoch_start,:] >= self.p_target)
                self.active_neur_start[j,k] = activity_start
                print(f"The number of active neurons at plateau start for replica {j}, layer {k} is {activity_start}")
                activity_end = np.sum(activity[epoch_end,:] >= self.p_target)
                self.change_active_neur[j,k] = activity_end-activity_start
                print(f"The change in number of active neurons over plateau for replica {j}, layer {k} is {activity_end-activity_start}")

    def compute_preactivation_rate(self): #NEED TO CHANGE THIS NAMING CONVENTION, ENSEMBLE_MEANS IS INCREDIBLY NONDESCRIPT
        """Calculates phi dot"""

        self.ensemble_means = {i:[] for i in range(self.numlayer)}
        self.ensemble_uncertainty = {i:[] for i in range(self.numlayer)}


        """Sanity checker"""
        for k in range(self.numlayer):
                    print(f'Layer {k}: ensemble_derivs[{k}] has {len(self.trial.ensemble_derivs[k])} entries, first entry shape: {self.trial.ensemble_derivs[k][0].shape}')

        for i in range(self.numlayer):
            print(f"On ensemble calculations for layer {i}")

            stacked_ensemble = np.stack(self.trial.ensemble_derivs[i], axis = 0) #Shape is now [ensemble, epochs-1,batch, neurons]
            """What is going on in above line:
            We are taking the FIRST ELEMENT of the ensemble_derivs, recall that this corresponds to selecting a specific layer
            We then stack all these list elements into one numpy array along the 0th axis of the components, i.e., along the ensemble_num axis
            Therefore we have one numpy array for each layer, of the shape [ensemble, epoch -1, batch, neuron ]
            """

            #Take the mean value with respect to the ensembles
            ens_mean = np.abs(np.mean(stacked_ensemble, axis = 0)) #New dimensionality is [epochs-1,batch, neurons]

            #Now take the mean over the remaining axes, i.e., batches first, then calculate error, then take remaining mean
            ensemble_means_neurons = ens_mean.mean(axis = 1)
            #Now have something of shape [epochs -1, neurons]

            # #Want to calculate the bootstrap uncertainty for this list
            # for j in range(ensemble_means_neurons.shape[0]):
            #     ensemble_uncertainty[i].append(Bootstrap_Analysis(ensemble_means_neurons[j,:]))
            """Above is commented out to save computation time"""

            self.ensemble_means[i] = np.mean(ensemble_means_neurons,axis =1) #Now averaging over all the neurons

            # results = Parallel(n_jobs= -1)(delayed(Bootstrap_Analysis)(ensemble_means_neurons[j,:]) for j in range(ensemble_means_neurons.shape[0])
            #                                )
            # ensemble_uncertainty[i] = np.array(results)
        
##########################################################
    """All functions below the check lines and the legacy section are yet to be updated as of September 17th"""

    def compute_output_deriv_vec(self):
        """ens_dout_dt_vec[k] represents the change from epoch k to epoch k+1"""

        # self.ens_dout_dt_vec = np.zeros((self.trial.epochs-1,len(self.trial.x_train)))
        # print(f"Shape of ensemble dout/dt matrix is {self.ens_dout_dt_vec.shape}")

        output_idx = self.numlayer-1 
        stacked_ensemble = np.stack(self.trial.ensemble_derivs[output_idx], axis = 0) #Shape is now [ensemble, epochs-1,batch, neurons]
        stacked = stacked_ensemble.squeeze(axis =3) #Remove the neuron dimension since only one output neuron
        self.ens_dout_dt_vec =np.mean(stacked,axis =0)
        #Expect dimensions to be [epoch-1, 16]
        print(f"Dimensionality of the ensemble average time derivative output vector is {self.ens_dout_dt_vec.shape}")
        # self.ens_dout_dt_vec_uncert = np.std(stacked,axis = 0)/np.sqrt(self.ensemble)

    def compute_ens_avg_residuals(self):
        """Computes the ensemble average residuals from the ensemble average eigenvalues and eigenvectors calculated in compute eigen_quant
          Recall that compute eigen_quant calculates the eigenvalues/vectors from the ensemble average NTK at each timestep"""
        alignint = self.trial.ntk_recorder.alignmentint
        epochs = self.trial.epochs


        self.ensemble_res = np.zeros((len(self.trial.ntk_recorder.recorded_epochs),self.e_amount))
        #Stores the individual residuals per epoch per eigenvector
        #has dimensionality [epochs, e_amount]

        self.ens_resXeval = np.copy(self.ensemble_res)
        #Stores the residual * corresponding eigenvalue


        self.dout_proj = np.zeros((len(self.trial.ntk_recorder.recorded_epochs[:-1]), self.e_amount)) 

        #Stores the projection of the output onto the eigenspace

        ens_out_avg = np.mean(self.trial.ntk_recorder.ensemble_output, axis =0) #Average output vector

        for i in range(self.ensemble_avg_NTK.shape[0]):

            epoch_value = self.trial.ntk_recorder.recorded_epochs[i] #Converting to the actual epoch, important if alignmentint is ever not 1

            eigspace = self.ensemble_evecs[i,:,:]

            output_vec = eigspace.T @ ens_out_avg[i,:] #eigenspace at time T dotted with the output at time T

            # print(f"Shape of output_vec is {output_vec.shape}")

            target_vec = eigspace.T@ self.y_train #Projection of the target function onto the eigenspace, how much of the target is in each eigendirection
            # print(f"Shape of target_vec is {target_vec.shape}")


            resid_signed = target_vec-output_vec #Keep  the sign identity for the resXeval calculations
            self.ensemble_res[i,:] = np.abs(resid_signed)
            #Each component of this vector gives how far off the model's output is from the target in each eigendirection
            #Note: under current configuration i.e., np.abs(), we simply store how far off the data is but not necessarily the direction (undershooting vs overshooting)
            self.ens_resXeval[i,:] = resid_signed * self.ensemble_evals[i,:]
            if (epoch_value + alignint < self.trial.epochs): #adding alignmentint checks if we have a forward prediction for this specific instance
                """
                Only do if  we actually have valid measurement (forward deriv does not work for final point)

                Output vector projected onto eigenspace:
                Output vectors are recorded at the start of the training loop(very first thing called is the NTK recorder), this means that 
                the output vector recorded for epoch k, is a forward prediction.  As a result, we should have dimensionality [epochs-1, e_amount], where 
                all values are stored except for the final (as we have no prediction for the final epoch), hence the epoch_value+1<self.trial.epochs
                As a result of np.diff, we only lose one epoch, hence why we do < and not <="""
                deriv_project = eigspace.T @ self.ens_dout_dt_vec[epoch_value,:]

                # print(f"Shape of the derivatie projection is {deriv_project.shape}")

                self.dout_proj[i,:] = deriv_project
                # print(f"The shape of the projection of the activation rate on the eigenvalues are {self.dout_proj.shape}")

        print(f"Shape of ensemble_res is{self.ensemble_res.shape}")
        print(f"Shape of ensemble_resXeval is{self.ens_resXeval.shape}")
        print(f"Shape of dout_proj is{self.dout_proj.shape}")

    def compute_relative_resXeval_size(self):
        """Compute_ens_avg_residuals must be ran prior to this 
        Computes the relative size of the residualXeval at each epoch
        """

        self.relative_resXeval = np.zeros_like(self.ens_resXeval)
        for i in range(self.ens_resXeval.shape[0]):
            sum_term = np.sum(np.abs(self.ens_resXeval[i,:]))
            if sum_term ==0:
                continue
            for j in range(self.ens_resXeval.shape[1]):
                append_item = np.abs(self.ens_resXeval[i,j]/sum_term)
                assert append_item <= 1 +1e-8, f"Relative resXeval append item epoch {i}, eval {j+1} is greater than 1"
                self.relative_resXeval[i,j] = append_item

    def compute_relative_res_size(self):
        """Compute_ens_avg_residuals must be ran prior to this 
        Computes the relative size of the residualXeval at each epoch
        """

        self.relative_res = np.zeros_like(self.ensemble_res)
        for i in range(self.ensemble_res.shape[0]):
            sum_term = np.sum(np.abs(self.ensemble_res[i,:]))
            if sum_term ==0:
                continue
            for j in range(self.ensemble_res.shape[1]):
                append_item = np.abs(self.ensemble_res[i,j]/sum_term)
                assert append_item <= 1 +1e-8, f"Relative res append item epoch {i}, eval {j+1} is greater than 1"
                self.relative_res[i,j] = append_item

    def individual_replica_TOI(self):
        """A function meant to calculate each replica's individual times of interest, NTKs at ti, etc.  
        to be used ideally in a hyperparameter scan over C_W

        Calculates times of interest and eigenvalues on a per replica basis, as opposed to doing so on ensemble wide NTK's
        """
        x_len = len(self.trial.x_train.squeeze())

        ensemble_t1_NTK = np.empty((self.ensemble,x_len,x_len))
        ensemble_t2_NTK = np.empty((self.ensemble,x_len,x_len))

        self.replica_timesOI_index_array = np.empty((self.ensemble,2)) #dimensionality [replicas, 2(t1,t2)]
        #array to store all the indices corresponding to NTK's at t1, t2
        for i in range(self.ensemble):
            initial_NTK = self.trial.ntk_recorder.ensemble_NTK[i,0,:]
            eval1 = np.linalg.eigvalsh(initial_NTK)[-1] #last eigenvalue corresponds to largest eigenvalue
            t1 = 1/eval1

            t1_index = self.convert_times(t1,training_time=True)

            if t1_index is None:
                ensemble_t1_NTK[i,:] = np.nan
                print("t1_end_index is none")
                #if t1 end index is beyond, so will t2 be
                ensemble_t2_NTK[i,:] = np.nan
                continue #continue so as not to terminate the loop

            self.replica_timesOI_index_array[i,0] = t1_index
            
            
            t1_NTK = self.trial.ntk_recorder.ensemble_NTK[i,t1_index,:]

            ensemble_t1_NTK[i,:] = t1_NTK

            eval2_at_t1 = np.linalg.eigvalsh(t1_NTK)[-2]

            t2 = 1/eval2_at_t1
            t2_index = self.convert_times(t2, training_time=True)

            self.replica_timesOI_index_array[i,1] = t2_index

            t2_NTK = self.trial.ntk_recorder.ensemble_NTK[i,t2_index,:]
            ensemble_t2_NTK[i,:] = t2_NTK

            print(f"Done with finding replica {i}'s NTKs")
        avg_t1_NTK = np.nanmean(ensemble_t1_NTK, axis =0)
        avg_t2_NTK = np.nanmean(ensemble_t2_NTK, axis = 0)
        ensemble_avg_lambda2 = np.linalg.eigvalsh(avg_t1_NTK)[-2]
        ensemble_avg_lambda3 = np.linalg.eigvalsh(avg_t2_NTK)[-3]
        print(f"Ensemble average eigenvalues are l2 = {ensemble_avg_lambda2}, and l3 = {ensemble_avg_lambda3}")

        ensemble_avg_t2 = 1/ensemble_avg_lambda2
        ensemble_avg_t3 = 1/ensemble_avg_lambda3

        return ensemble_avg_t2,ensemble_avg_t3, ensemble_avg_lambda2,ensemble_avg_lambda3 

    def compute_replica_plateau_evec_rotations(self, target_eval =3):
        """Function meant to calculate the eigenvector rotations during plateau phase relative to 
        the eigenvector orientation at plateau start.  It is believed that the rotation in this learning phase 
        is much subtler than in previous phases or in the subsequent descent from the saddle.

        This is done on a replica by replica-basis so as to avoid losing data from non-uniform plateau start/stop
        """
        print("Computing replica evec3 rotations")
        self.replica_plateau_rotations = {}
        self.replica_plateau_rotations_times ={}

        recorded_plateau_times = self.trial.plateau_recorder.plateau_bounds
        for i in range(self.ensemble):
            if np.isnan(recorded_plateau_times[i,1]):
                print(f"replica {i} does not converge")
                continue #go to next iteration
            else:
                rotations_list = []
                # plat_start_NTK_timeindex = self.convert_times(time = recorded_plateau_times[i,0])
                # plat_end_NTK_timeindex = self.convert_times(time = recorded_plateau_times[i,1])
                start_time = self.convert_times(recorded_plateau_times[i,0])
                end_time = self.convert_times(recorded_plateau_times[i,1])
                # print(f"replica {i}: start_time={start_time} (type={type(start_time)}), end_time={end_time} (type={type(end_time)})")
                if start_time is None or end_time is None:
                    print(f"replica {i} plateau bounds fall outside recorded epoch range, skipping")
                    continue
                plateau_duration = np.arange(start_time, end_time)
                # print(f"replica {i}: plateau_duration dtype={plateau_duration.dtype}, len={len(plateau_duration)}, sample={plateau_duration[:5]}")


                #logger.debug stuff
                # print(f"Plateau start for replica {i} is {recorded_plateau_times[i,0]}")
                # print(f"First 5 plateau_duration values are {plateau_duration[:4]}")

                self.replica_plateau_rotations_times[i] = np.array(self.trial.ntk_recorder.recorded_epochs)[plateau_duration].tolist()

                #Only concerned with the replicas that manage to leave their own plateau
                initial_NTK = self.trial.ntk_recorder.ensemble_NTK[i,0,:]
                eval1 = np.linalg.eigvalsh(initial_NTK)[-1] #last eigenvalue corresponds to largest eigenvalue
                t1 = 1/eval1
    
                t1_index = self.convert_times(t1,training_time=True)
                t1_NTK = self.trial.ntk_recorder.ensemble_NTK[i,t1_index,:]

                eval2_at_t1 = np.linalg.eigvalsh(t1_NTK)[-2]
                t2 = 1/eval2_at_t1
                t2_index = self.convert_times(t2, training_time=True) 

                t2_NTK =self.trial.ntk_recorder.ensemble_NTK[i,t2_index,:]

                discard, evecs = np.linalg.eigh(t2_NTK)
                evec3_t2 = evecs[:,-target_eval]
                evec3_t2_norm = np.linalg.norm(evec3_t2)
                for idx in plateau_duration:
                    # print(epoch)
                    ntk_epoch = self.trial.ntk_recorder.ensemble_NTK[i,idx,:] 
                    #should just be able to index the epoch directly since the alignmentint has been accounted for
                    discard, evecs = np.linalg.eigh(ntk_epoch)
                    evec3 = evecs[:,-target_eval]
                    angle_of_rotation = self.rotation_help(evec3_t2,evec3,evec3_t2_norm)
                    rotations_list.append(angle_of_rotation)
                self.replica_plateau_rotations[i] = rotations_list
            print(f"Done with replica {i}")
            # print(self.replica_plateau_rotations.keys())

    def compute_ensemble_evec3_plateau_rotate(self, target_eval =3):
        """Computes the ensemble average rotation of eigenvector 3 as compared to eigenvector 3 at plateau start 
        throughout the plateau region.
        
        Requires: 
        compute_eigen_quant
        compute_timesOI
        """      
        print("Computing ensemble evec3 rotation")
        ens_mean_plat_times = np.nanmean(self.trial.plateau_recorder.plateau_bounds, axis=0)

        mean_t2 = ens_mean_plat_times[0]
        mean_t3 = ens_mean_plat_times[1]
        # print(f"mean t_3 is {mean_t3}")

        self.ens_plateau_evec3_rotations =[] 

        ens_t2_converted = self.convert_times(mean_t2)
        ens_t3_converted = self.convert_times(mean_t3)

        # print(f"ens_t3_converted is {ens_t3_converted}")
        # print(f"ens_t2_converted is {ens_t2_converted}")

        plateau_duration = np.arange(ens_t2_converted, ens_t3_converted)


        # print(f" plateau_duration dtype={plateau_duration.dtype}, len={len(plateau_duration)}, sample={plateau_duration[:5]}")


        self.plateau_ntk_times = np.array(self.trial.ntk_recorder.recorded_epochs)[plateau_duration].tolist()
        # print(self.plateau_ntk_times)
        ens_NTK_t2 = self.ensemble_avg_NTK[ens_t2_converted,:]
        discard, evecs = np.linalg.eigh(ens_NTK_t2)
        evec3_t2 = evecs[:,-target_eval]
        evec3_t2_norm = np.linalg.norm(evec3_t2)
        for idx in plateau_duration:
            ntk_epoch = self.ensemble_avg_NTK[idx,:]

            discard, evecs = np.linalg.eigh(ntk_epoch)
            evec3 = evecs[:,-target_eval] 

            rotation_angle = self.rotation_help(evec3_t2,evec3,evec3_t2_norm)
            self.ens_plateau_evec3_rotations.append(rotation_angle)


#################################################
    """Still need to fix these"""
    def compute_residuals_per_replica(self):

        """Outdate function, we want the eigenvalues/vectors of the ENSEMBLE NTK, not the ensemble average nth eigenvector"""
        
        target = self.trial.y_train.squeeze().numpy()
        # print(f"Shape of target vector is {target.shape} ")
        ensemble_res = np.zeros((self.ensemble, len(self.trial.ntk_recorder.recorded_epochs),len(self.trial.x_train)))
        #Dim [Ensemble, epoch, vector length]
        ensemble_resXeval = np.copy(ensemble_res)


        num_epochs_deriv = np.copy(ensemble_res)
        #Note, the activation vector is calculated once every epoch, but the NTK's are recorded every AlignmentInt, thus care has to be taken when calculating further down

        dout_proj = np.zeros((self.ensemble, len(self.trial.ntk_recorder.recorded_epochs[1:]), len(self.trial.x_train)))
        outer_idx = self.numlayer-1
        for i in range(self.ensemble):
            """There is a chance that eigenvectors/eigenvalues are very close and ill defined beyond eigenvalue 2, as such, more care is needed to ensure that statistical
            fluctuations are not the cause of the fixed behavir between residuals and rotations.
            
            Currently projecting over the WHOLE eigenspace, not just the selected eigenvectors eigenspace
            """
            for j in range(len(self.trial.ntk_recorder.recorded_epochs)):
                epoch_value = self.trial.ntk_recorder.recorded_epochs[j] #Need to convert to actual epoch to get the correct activation rate vector

                evals, evecs = np.linalg.eigh(self.trial.ntk_recorder.ensemble_NTK[i,j,:,:])
                #Grabbed ith replicas NTK at time j, should have remaining dimensionality [16 X 16]

                output = self.trial.ntk_recorder.ensemble_output[i,j,:]

                idx = np.argsort(evals)[::-1]
                for z in range(len(target)):
                    vec = evecs[:,z]
                    sign = np.sign(vec[np.argmax(np.abs(vec))])
                    evecs[:,z] *=sign
                eigspace = evecs[:,idx]
                evals_sorted = evals[idx]
                output_vec = eigspace.T @ output
                target_vec = eigspace.T @ target

                abs_res = np.abs(target_vec-output_vec)
                ensemble_res[i,j,:] = abs_res
                ensemble_resXeval[i,j,:] = abs_res *evals_sorted

                if j>0:
                    """Also want to take the output vector on a replica by replica basis
                    Recall dimensionality of 
                    """
                    

                    dout_proj[i,j-1,:]= eigspace.T @self.ens_dout_dt_vec[epoch_value-1,:]
                    #Need to offset epoch_value by -1 to account for initial 0th epoch

        self.ensemble_res = np.mean(ensemble_res,axis = 0)
        print(f"The shape of ensemble average residuals is {self.ensemble_res.shape}")

        self.ens_resXeval = np.mean(ensemble_resXeval, axis =0)
        print(f"The shape of ensemble average residuals * evals is {self.ensemble_res.shape}")

        self.ens_dout_proj = np.mean(dout_proj, axis = 0)
        print(f"The shape of ensemble average activation rate projection is {self.ensemble_res.shape}")

    def compute_residuals_static(self):
            #First need to find ensemble averages of the NTK's throughout training, as well as the model outputs
            target = self.trial.y_train.squeeze().numpy()
            # print(f"Shape of target vector is {target.shape} ")
    
            self.ensemble_res_static = np.zeros((len(self.trial.ntk_recorder.recorded_epochs),len(self.trial.x_train)))
            #One per epoch, for each point
            #Need average ensemble output per epoch
    
            ens_out_av = np.mean(self.trial.ntk_recorder.ensemble_output, axis =0)
    
            ens_avg_NTK = np.mean(self.trial.ntk_recorder.ensemble_NTK, axis = 0)

            eigvals, eigvecs = np.linalg.eigh(ens_avg_NTK[0,:])
            idx = np.argsort(eigvals)[::-1]
            eigspace = eigvecs[:,idx]

            # print(f'Shape of ens_avg_NTK is {ens_avg_NTK.shape}')
            for k in range(ens_avg_NTK.shape[0]):
                evals, evecs = np.linalg.eigh(ens_avg_NTK[k,:])
                output_vec = eigspace.T @ ens_out_av[k,:] #eigenspace at time T dotted with the output at time T

                # print(f"Shape of output_vec is {output_vec.shape}")

                target_vec = eigspace.T@ target #Projection of the target function onto the eigenspace, how much of the target is in each eigendirection

                self.ensemble_res_static[k,:] = np.abs(target_vec-output_vec)
                #Each component of this vector gives how far off the model's output is from the target in each eigendirection
                #Note: under current configuration i.e., np.abs(), we simply store how far off the data is but not necessarily the direction (undershooting vs overshooting)
#################################################

    """Below are legacy functions, unlikely to be utilized (and consequently unoptimised), however still helpful for certain cases.  
    Assume they are not up to date with current state of code"""

    def compute_chi(self):
        """Chi array will have one less value than ensemble, since layer 1 will not have a chi reading,
        
        compute_preactivation_rates MUST be called before calling compute_chi
        """

        self.chi_array = {i:[] for i in range(1,self.numlayer)}
        """Calculating chi defined as ratio of pre-activation derivs"""
        for k in range(1,self.numlayer):
            ############################################################################
            if k ==self.numlayer:
                self.chi_array[k] = ((self.ensemble_means[k]/self.ensemble_means[k-1]))/10
                """Chi is dependent on number of neurons in layer, last output has 1/10th neurons so need to scale accordingly""" 
                """Should think of a better way to do this"""
            ############################################################################
            else:
                self.chi_array[k] = ((self.ensemble_means[k]/self.ensemble_means[k-1]))
            print(f"Length element is {len(self.ensemble_means[k]/self.ensemble_means[k-1])}") 

        print(f"The length of a chi list in the chi array is {len(self.chi_array[1])}")
   
    def export_onnx(self): #Don't yet know where to put this one
        """Exports the model to an onnx file"""

        inst_input = torch.randn((1,1)).double() 
        """Since torch.export will run a tracer through the model, it does not actually matter what the input data is
        To make it as simple as possible, just using a 1X1 torch tensor
        """
        onnx_program = torch.onnx.export(self.model,inst_input)
        onnx_program.save(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\MLP_model.onnx')
        print(f"The model has been exported to an onnx file")

    def compute_timesOI(self,number_of_times = 3):
        """Function is meant to compute the times of interest from NTK theory.
        NTK theory suggests that the relevant timescales to the learning process are as follows:
        [0,t1], where t1 = 1/lambda1(0),
        [t1, t2], where t2 = 1/lambda2(t1),
        [t2,t3], where t3 = 1/lambda3(t2),
        etc

        To confirm that this behavior is observed experimentally, these times must be calculated on a replica by replica basis; as it has been observed that
        plateau length is not universal, and is highly tied to initilisation, any sort of ensemble average would be inaccurate (similar to the plateau end identification)  
        """
        self.times_OI =np.empty((self.ensemble,number_of_times)) #dimensionality is [ensemble, number of times of interest]

        initialNTKs = self.trial.ntk_recorder.initial_NTK

        ens_NTK = self.trial.ntk_recorder.ensemble_NTK

        recorded_NTKtimes = self.trial.ntk_recorder.recorded_epochs

        self.eval3_at_t2 = []
        self.eval2_at_t1 = []

        for i in range(self.ensemble): #iterate over ensemble members        
            eval1 = np.linalg.eigvalsh(initialNTKs[i])[-1] #np.linalg.eigvalsh returns eigenvalues in ascending order, want largest so last component
            # print(f"eval 1 for replica {i} is {eval1}")
            t1 = 1/eval1
            # print(f"T1 calculated for replica {i} is {t1}")
            self.times_OI[i,0] = t1

            #Now for eigenvalue 2
            t1_end_index = self.convert_times(t1,training_time=True) #inverse eigvals convert to training time, not epochs

            """Checking if the convert times has returned None, if it has, assign all further times of interest to be np.nans, which will then be filtered"""
            if t1_end_index is None:
                self.times_OI[i,1:] = np.nan
                print("t1_end_index is none")
                continue #continue so as not to terminate the loop


            eval2 = np.linalg.eigvalsh(ens_NTK[i,t1_end_index,:,:])[-2]
            self.eval2_at_t1.append(eval2)
            # print(f"eval 2 for replica {i} is {eval2}")
            t2 = 1/eval2
            print(f"T2 calculated for replica {i} is {t2}")
            self.times_OI[i,1] = t2

            if number_of_times>2:
                #adds in possibility of just checking for first two times of interest, 
                #Eigenvalue 3
                t2_end_index = self.convert_times(t2,training_time=True)
                eval3 = np.linalg.eigvalsh(ens_NTK[i,t2_end_index,:,:])[-3]
                # print(f"eval 3 for replica {i} is {eval3}")
                self.eval3_at_t2.append(eval3)
                t3 = 1/eval3
                # print(f"T3 calculated for replica {i} is {t3}")
                self.times_OI[i,2] = t3

                print(f"Evals for replica {i} are {eval1}, {eval2}, and {eval3}")
                print(f"Times for replica {i} are {t1}, {t2}, and {t3}")
                # print(f"Actual epoch times for replica {i} are {}")

                """Testing eval 3 stability """
                # for idx in range(0, 10):
                #     evals = np.linalg.eigvalsh(ens_NTK[i, idx, :, :])
                #     print(f"replica {i} epoch {idx*self.trial.ntk_recorder.alignmentint}: eval3 = {evals[-3]}")

        #want an unfiltered version of the times_OI list
        self.times_OI_unfiltered = np.copy(self.times_OI)
        
        max_time = self.trial.epochs *self.trial.lr
        self.times_OI[self.times_OI > max_time] = np.nan

    def compute_phi_neuron(self):
        """self.phi has the following dimensionality
        Self.phi is a dictionary with amount of lists = numlayers
        each list contains numpy arrays with shape [epochs,batch/data,neuron], corresponding to the activity in the selected layer for corresponding epochs etc

        We will stack these lists into one array, so the dimensionality will be [replica, epochs, batch/data, neuron], then we can simply calculate the mean and std
        
        
        NOTE, NO LONGER WORKING!!!
        
        """

        self.phi_neuron = {i:[] for i in range(self.numlayer)}
        self.phi_sq = {i:[] for i in range(self.numlayer)}

        #self.phi_sq_uncert = {i:[] for i in range(self.numlayer)}
        for k in range(self.numlayer):
            stacked_ensemble = np.stack(self.trial.phi[k], axis =0) #Dimensionality [replica, epoch, batch, neuron]
            stacked_ensemble = np.abs(stacked_ensemble) #Made it absolute value here
            ensemble_mean = np.mean(stacked_ensemble, axis =0) #Dim is [epoch, batch, neuron]
            #now need to take mean over remaining dimensions
            self.phi_neuron[k] = (np.mean(ensemble_mean, axis = 1)) #We are left with a list of dim [epoch, neuron]

            squared_ensemble = np.stack(self.trial.phi[k], axis =0) #Dimensionality [replica, epoch, batch, neuron]
            squared_ensemble = squared_ensemble**2 #Recall we are interested in the square
            squared_mean = np.mean(squared_ensemble, axis =0) #Dim is [epoch, batch, neuron]
            #now need to take mean over remaining dimensions
            self.phi_sq[k] = (np.mean(squared_mean, axis = (1,2)))

    def compute_delta_weight_times(self, snapshots =5): #TIME CVONVERSION IS INCORRECT
        """Computes the difference in weight values along intervals specified by the times_of_int linspace calculator
        The difference is a simple forward finite difference schema, hence why we set endpoint = False as there is no forward derivative here
        These values are meant to be used for histogram and heatmap analysis of how the weights change during training
        """
        times_of_int = np.linspace(0,self.trial.epochs,snapshots, endpoint = False) #this is in raw epochs, do not want to include the last "epoch", since it is out of bounds
        alignmentint = self.trial.ntk_recorder.alignmentint
        x=times_of_int/alignmentint
        self.dw_times = (alignmentint* np.round(x)).astype(int)#must be integers to use as indices

        self.delta_weight_dict = {} #Don't need one for the first hidden layer as the input is size 1, so weight tensor is a 1x10
        for i in range(1,self.param_dict_length): #iterate over layers of interest
            weight_array = self.trial.parameter_recorder.ensemble_weights[i]
            #Remaining dimensionality is [replica][epoch][weight tensor], slice along the selected times 
            sliced_weights = weight_array[:,self.dw_times,:,:] 

            delta_sliced_weights = np.diff(sliced_weights, axis = 1) #expect something of shape [ensemblenum, snapshots-1, 10,10]
            print(f"The delta sliced weight tensor is of shape {delta_sliced_weights.shape}")
            self.delta_weight_dict[i] = delta_sliced_weights

    def compute_ens_dweight_plateau(self):
        """Calculates difference in weight across all replicas from plateau start to plateau end.  This calculation is done on a per replica basis, as plateau
        end (and to some degree start) is not universal.  This data is meant to be utilized in a histogram plot; a heatmap would not be of particular help here 
        as it is expected that which neurons are actually contributing varies on a per replica basis, and any ensemble average would flatten the behavior."""

        alignmentint = self.trial.ntk_recorder.alignmentint

        self.ens_dweight_hist_vals = {k:[] for k in range(1,self.param_dict_length)} #want a dictionary here, as we will need one per hidden layer 

        for i in range(1,self.param_dict_length): #layers 2-3
            weight_array = self.trial.parameter_recorder.ensemble_weights[i] #Grabs the ith layer of the weight recorder
            #Remaining dimensionality is [replica][epoch][weight tensor], grab each replica, and then the difference over plateau start and end
            for j in range(self.ensemble):
                plateau_bounds =self.ens_plateau_bounds_post[j] 
                if not np.isnan(plateau_bounds[-1]): #Checks to see if the plateau end has been recorded for this particular replica 
                    """If no nans are present, we can proceed with the calculations, but first the times must be converted into their respective alignment times
                    want to go from the raw epoch value, into the closes alignment int value (i.e., the closest time to recorded plateau start/end that the weights were
                    recorded.)
                    """

                    #############################################################################
                    times_of_int = np.array(plateau_bounds)
                    print(times_of_int)
                    x=times_of_int/alignmentint
                    print(x)
                    dw_times = (np.round(x)).astype(int)#must be integers to use as indices
                    print(dw_times)
                    #############################################################################
                    #Above is technically wasted computation time, since the times are same per layer, may be better to restructure outside the loop

                    epoch_start = dw_times[0]
                    epoch_end = dw_times[-1]
                    weight_start = weight_array[j,epoch_start,:,:]
                    weight_end = weight_array[j,epoch_end,:,:]

                    delta_weight_flat = (weight_end-weight_start).ravel() #need to flatten it to make it easier to plot with ax

                    self.ens_dweight_hist_vals[i].append((delta_weight_flat.tolist()))

                else:
                    pass

    def compute_param_deriv(self):#STILL NEEDS ABS REL CHANGE
        """Want a list, one for each layer
        In each list will be each replica's """
        self.ens_weight_deriv = {i:[] for i in range(self.param_dict_length)}
        self.ens_bias_deriv = {i:[] for i in range(self.param_dict_length)}
        self.ens_weight_abs_rel_change = {i:[] for i in range(self.param_dict_length)}
        for i in range(self.param_dict_length):
            layer_param_array = self.trial.parameter_recorder.ensemble_weights[i]
            #Will get something out of shape [replicas, epochs, input,output]

            layer_bias_array = self.trial.parameter_recorder.ensemble_biases[i]
            #Something of shape [replicas,epochs,width]
            for j in range(self.ensemble):
                replica = layer_param_array[j]
                inst_deriv = np.diff(replica, axis = 0) #Should grab the jth replica, and take the deriv along the 0th axis, which would now be epoch
                # print(f"For layer {i+1}, replica {j} dimensionality before was {layer_param_array[j].shape}, dimensionality after is {inst_deriv.shape}")
                #Expect to see [epoch,input,output], [epoch-1,input,output]
                self.ens_weight_deriv[i].append(inst_deriv.tolist())

                #now doing the biases
                replica_biases = layer_bias_array[j]
                inst_bias_deriv = np.diff(replica_biases,axis =0)
                # print(f"For layer {i+1} replica {j}, dim before was {replica_biases.shape}, dim after is {inst_bias_deriv.shape}")
                self.ens_bias_deriv[i].append(inst_bias_deriv.tolist())



        for i in range(self.param_dict_length):
            self.ens_weight_deriv[i] = np.array(self.ens_weight_deriv[i])
            self.ens_bias_deriv[i] = np.array(self.ens_bias_deriv[i])

    def compute_NTK_pts(self):
        N = self.trial.x_train.shape[0]
        self.NTK_points = 1/self.mean_eigenvals 
        self.NTK_point_uncert = 1*self.mean_eigenvals_std/(self.mean_eigenvals**2)

        """Filtering out any eigenvalues which are negative, this is almost always due to floating point instability"""
        if np.any(self.mean_eigenvals< 0):
            print(f"Negative eigenvalue in list")
        print(f'The eigenvalues of the ensemble average NTK matrix are {self.mean_eigenvals} ')
        print(f"The specific points of interest from the initial NTK axes is {self.NTK_points}")

        """Need to remove any points which are outside the maximum epoch range"""
        max_time = self.trial.epochs * self.trial.lr #Adjust it to be in terms of learning rate
        self.NTK_points = self.NTK_points[self.NTK_points <= max_time].real #Only care about real eigenvalues, filter out any points which are beyond our training range
        self.NTK_points = self.NTK_points[self.NTK_points >0]
