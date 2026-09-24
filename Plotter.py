
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math as m
import networkx as nx
import seaborn as sns
from pathlib import Path


"""In the readme should explain the need for the eigenval pair"""
class Plotter:
    def __init__(self,data,trial,regions,performance_snapshots,SaveFig, output_dir,show,nodesize, edgesize, groupings = 2):
        self.data = data
        self.trial = trial
        self.regions = regions

        
        self.lr = self.trial.lr
        """Needed for converting to right times"""

        self.output_dir = Path(output_dir)

        self.savefig = SaveFig

        self.show = show

        """This requires the phi_target times are computed before we initialise the plotter object"""
        # self.phi_colors = plt.cm.tab10(np.linspace(0, 1, len(self.data.phi_target_times)))

        self.train_time_total = self.trial.train_time_total

        #Will be used for the visuals
        self.nodesize = nodesize
        self.edgesize = edgesize

        self.groupings = groupings

        self.convert_times() #NEEDs to be called

    """Enclosed within the # lines below are helper functions utilized by the plotter class"""
######################################################################
    def convert_times(self):#Should change some of the naming conventions here
        """Helper function to convert the trial's stored lists of relevant epochs into lists of training times"""
        
        """Still unsure of activation"""
        # self.train_time_activation = self.trial.recorded_epochs_activation *self.lr
        self.train_time_activation = np.array(self.trial.recorded_epochs_activation[1:]) * self.lr
        """To be used for the pre-activation derivatives, hence the 1 offset, because when we take forward derivative we have no information for the first value"""

        self.train_time_activation_roc = np.array(self.trial.recorded_epochs_activation_roc) *self.lr
        self.train_time_alignment = np.array(self.trial.ntk_recorder.recorded_epochs) *self.lr

        self.train_time_loss = np.array(self.trial.loss_recorder.recorded_epochs) * self.lr
        # inst_loss = self.trial.loss_recorder.recorded_epochs.pop(0)
        # self.train_time_dl = np.array(inst_loss) *self.lr
        self.train_time_dl = self.train_time_loss[1:]
        self.train_time_rotation = np.array(self.trial.ntk_recorder.recorded_epochs) * self.lr

        self.train_time_record = np.array(self.trial.recorded_epochs_activation) * self.lr
        """Total epochs where activation is recorded, to be used for phisq plots"""

        self.performance_times = np.array(self.trial.performance_recorder.recorded_epochs) * self.lr 

    def get_mask(self, time_array, time_start,time_end):
        """Helper function to effectively filter to only regions of interest for plots"""
        return (time_array >= time_start) & (time_array <= time_end)    
    
    def add_NTK_points_regions(self, figure,t_start,t_end):
        for j in range(len(self.data.NTK_points)):
            if t_start<= self.data.NTK_points[j] <= t_end:
                figure.axvline(self.data.NTK_points[j], color = 'purple', alpha = 0.3)

    def add_NTK_points(self, figure):
        for j in range(len(self.data.NTK_points)):
            figure.axvline(self.data.NTK_points[j], color = 'purple', alpha = 0.3)

    def plot_save_or_show(self, fig, name):
        """A helper function to be called after creating a figure in any of the below functions.  Determines what to do with the created figure"""
        if self.savefig:
            fig.savefig(self.output_dir / f"{name}.png")
        if self.show:
            plt.show()
        plt.close(fig)

    def _windows(self,regions):
        """Creates the windows which certain plotter functions can utilize to break up the plot into specific regions of interest.
        This is an option primarily as the scale of specific datum are rather extreme, and to include maximums, we lose a lot of 
        fidelity in plotting over the whole timeframe.  To give a concrete example, plotting the ensemble activity per layer, we lose a lot of
        small fluctuations in the plateau region as a result of the scale of the activity at beginning and end.  
          
        Meant to be passed a list of regions created from the make_regions function in the Run_File
          """
        if regions is None:
            return [(None, None, "")]
        return [(t_start,t_end, f"_{i}_{i+1}") for i, (t_start, t_end) in enumerate(regions)]


######################################################################

    def plot_abs_rel_change_regions(self):
        t_iter=0
        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_activation_roc, t_start, t_end)
            t = self.train_time_activation_roc[mask]
            plt.figure(figsize=(10,6))
            plt.plot(t, self.data.eval1_rateofchange_mean[mask], label = "Eigenvalue 1 rate of change")
            plt.plot(t, self.data.eval2_rateofchange_mean[mask], label = "Eigenvalue 2 rate of change")
            plt.plot(t, self.data.eval3_rateofchange_mean[mask], label = "Eigenvalue 3 rate of change")
            plt.fill_between(t, (self.data.eval1_rateofchange_mean+ self.data.eval1_rateofchange_uncert)[mask], 
                            (self.data.eval1_rateofchange_mean -self.data.eval1_rateofchange_uncert)[mask], alpha = 0.3)
            plt.fill_between(t, (self.data.eval2_rateofchange_mean+ self.data.eval2_rateofchange_uncert)[mask], 
                            (self.data.eval2_rateofchange_mean -self.data.eval2_rateofchange_uncert)[mask], alpha = 0.3)
            plt.fill_between(t, (self.data.eval3_rateofchange_mean+ self.data.eval3_rateofchange_uncert)[mask], 
                            (self.data.eval3_rateofchange_mean -self.data.eval3_rateofchange_uncert)[mask], alpha = 0.3)
            plt.legend()
            plt.ylabel(f"Absolute relative rate of change of eigenvalue")
            plt.xlabel(f"Training time")
            plt.title(f"Absolute relative rate of change of eigenvalues 1, 2 and 3 as they vary with training time")
            if self.savefig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\RelAbsChange_{t_iter}_{t_iter+1}.png')
            if self.show:
                plt.show()
            plt.close()
            t_iter+=1

    def plot_abs_rel_change(self):
            t = self.train_time_activation_roc
            plt.figure(figsize=(10,6))
            plt.plot(t, self.data.eval1_rateofchange_mean, label = "Eigenvalue 1 rate of change")
            plt.plot(t, self.data.eval2_rateofchange_mean, label = "Eigenvalue 2 rate of change")
            plt.plot(t, self.data.eval3_rateofchange_mean, label = "Eigenvalue 3 rate of change")
            plt.fill_between(t, (self.data.eval1_rateofchange_mean+ self.data.eval1_rateofchange_uncert), 
                            (self.data.eval1_rateofchange_mean -self.data.eval1_rateofchange_uncert), alpha = 0.3)
            plt.fill_between(t, (self.data.eval2_rateofchange_mean+ self.data.eval2_rateofchange_uncert), 
                            (self.data.eval2_rateofchange_mean -self.data.eval2_rateofchange_uncert), alpha = 0.3)
            plt.fill_between(t, (self.data.eval3_rateofchange_mean+ self.data.eval3_rateofchange_uncert), 
                            (self.data.eval3_rateofchange_mean -self.data.eval3_rateofchange_uncert), alpha = 0.3)
            plt.legend()
            plt.ylabel(f"Absolute relative rate of change of eigenvalue")
            plt.xlabel(f"Training time")
            plt.title(f"Absolute relative rate of change of eigenvalues 1, 2 and 3 as they vary with training time")
            if self.savefig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\RelAbsChange.png')
            if self.show:
                plt.show()
            plt.close()


    def plot_preactivations(self, regions = None, exclude_outer = False, NTK_Points = False):
        """A function to plot the ensemble average rate of change of pre-activations as it depends on time.  NTK theory states there is a 
        1/n suppression factor for the output of any given layer, where n is the width of the layer; hence the output layer is on average 10X 
        the size of all the other layers.  As such, there is an option to exclude the output, and plot all the hidden layers."""

        n_layers = len(self.data.ensemble_means) - (1 if (exclude_outer is True) else 0)
        for t_start, t_end, suffix in self._windows(regions):
            if t_start is None:
                mask = slice(None) #if there is no t_start we take the slice over the whole array --> No masking
            else:
                mask = self.get_mask(self.train_time_activation, t_start, t_end)

            fig, ax = plt.subplots(figsize = (10,6))
            for k in range(n_layers):
                ax.plot(self.train_time_activation[mask],self.data.ensemble_means[k][mask],label = f"Layer {k+1}")

            if NTK_Points is True:
                if t_start is None:
                    self.add_NTK_points(ax)
                else:
                    self.add_NTK_points_regions(ax,t_start,t_end)

            ax.set_xlabel('Training Time')
            ax.set_ylabel('Pre-Activation Derivative')
            ax.set_title('Finite Difference of Layer Pre-Activations')
            ax.legend()
            name = f"Activity{'NoOuter' if (exclude_outer is True) else ''}{suffix}"
            self.plot_save_or_show(fig, name)

    def plot_eigenvalues(self, NTK_Points = False):
        """Plots all calculated eigenvalues over the total time domain.  It is recommended to use eval_pairs over this function so as to better track
          potential eigenvalue overlapping, however there is nothing preventing this from working  """

        fig, ax = plt.subplots(figsize = (10,6))
        for k in range(self.data.ensemble_evals.shape[1]):
            ax.plot(self.train_time_alignment,self.data.ensemble_evals[:,k], label = f"Eigenvalue {k+1}")

        if NTK_Points is True:
            self.add_NTK_points(fig)
        # self.add_Phi_points(plt)           
        ax.set_xlabel(f"Training time")
        ax.set_ylabel(f" Eigenvalue")
        ax.set_title(f"Largest vs Training Time")
        plt.legend()
        self.plot_save_or_show(fig,"AllEvals")

    def plot_performances(self):
        """Plots the ensemble average performance (outputted points), at times specified by the performance tracker in the 
        Trial_Recorders_Model file.  Also plots the uncertainty, the training points, and the true values."""
        x_vals = (self.trial.x_test.squeeze() * (4/3)*m.pi).numpy()
        x_train_vals = (self.trial.x_train.squeeze() * (4/3)*m.pi).numpy()
        for k in range(self.data.ens_avg_perf.shape[0]):
            epoch_value = self.performance_times[k]
            fig, ax = plt.subplots(figsize= (10,6))
            ax.plot(x_vals, self.data.ens_avg_perf[k,:], label = f'Predicted Values')
            ax.plot(x_vals,self.trial.y_test.numpy(), label = f'True Values')
            # plt.scatter(self.x_train,self.y_train,label = "Training points")
            ax.scatter(x_train_vals, self.trial.y_train, label="Training points")
            ax.fill_between(x_vals, self.data.ens_avg_perf[k,:]+self.data.ens_perf_uncert[k,:],
                             self.data.ens_avg_perf[k,:]-self.data.ens_perf_uncert[k,:],color = 'blue', alpha = 0.3)
            #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()
            ax.set_xlabel(f'X')
            ax.set_ylabel(f'Y')
            ax.set_title(f'Performance of the model at training time {epoch_value}')
            ax.legend()
            self.plot_save_or_show(fig, f"Performance{k+1}")
            
    def plot_evec_rotations(self, regions = None, NTK_Points = False):
        """Plots the eigenvector rotations from their direction at initialisation.  Can do this over the whole time domain (default case) 
        or in specified regions. 

        If NTK_Points set to true, plots the inverse eigenvalues at initialisation
        """
        for t_start, t_end, suffix in self._windows(regions):
            if t_start is None:
                mask = slice(None) #if there is no t_start we take the slice over the whole array --> No masking
            else:
                mask = self.get_mask(self.train_time_rotation, t_start, t_end)
            t = self.train_time_rotation[mask]     

            fig, ax = plt.subplots(figsize = (10,6))
            for k in range(self.data.evec_rotation_mean.shape[1]):
                ax.plot(t,self.data.evec_rotation_mean[mask][:,k], label = f"Eigenvector {k+1}")

                """There is currently no eigenvector rotation uncertainty, to do so would require bootstrapping the ensemble ntk at each timestep which is
                quite computationally expensive.  As such, the functionality has been commented out, but remains able to be re-implemented should the uncertainty
                be calculated"""
                # ax.fill_between(t, self.data.evec_rotation_mean[mask][:,k] + self.data.evec_rotation_uncert[mask][:,k],
                #                 self.data.evec_rotation_mean[mask][:,k] - self.data.evec_rotation_uncert[mask][:,k], alpha = 0.3 )

            if NTK_Points is True:
                if t_start is None:
                    self.add_NTK_points(fig)
                else:
                    self.add_NTK_points_regions(fig, t_start, t_end)           

            ax.set_xlabel(f"Training time")
            ax.set_ylabel(f" Absolute Angle of Rotation")
            ax.set_title(f"Eigenvector Rotation vs Training Time")
            ax.legend()
            self.plot_save_or_show(fig,f"EvecRotate{suffix}")

    def plot_residuals(self):
        """Plots the residual of the target data and output data which have been projected onto an eigenvector of the NTK at a given time """
        fig, ax = plt.subplots(figsize = (10,6))
        for k in range(self.data.e_amount):
            ax.plot(self.train_time_alignment, self.data.ensemble_res[:,k], label = f"eval {k+1}") #kth vector component residual at all epochs
        self.add_NTK_points(ax)
        ax.set_xlabel('Training Time')
        ax.set_ylabel('Target Projection - Output Projection')
        ax.set_title(f'Residuals of Projections of Target Data and Output Data on Eigenspace')
        ax.legend()
        self.plot_save_or_show(fig, "Residuals")

    def plot_residuals_static(self):
        """Plots the residuals of the projections of target data and output data onto a static eigenspace (eigenspace taken at t=0).  This is done
        so to ensure/check that residuals do in fact go to zeroas time increases.  In the nonstatic case, since the whole eigenspace is rotating, it is possible
        (and based on experimental observation, likely) for residuals to increase after having decreased, or to simply not trail towards 0.
        """
        fig, ax = plt.subplots(figsize = (10,6))
        for k in range(self.data.e_amount):
            ax.plot(self.train_time_alignment, self.data.ensemble_res_static[:,k], label = f"eval {k+1}") #kth vector component residual at all epochs
        self.add_NTK_points(ax)
        ax.set_xlabel('Training Time')
        ax.set_ylabel('Target Projection - Output Projection')
        ax.set_title(f'Residuals of Projections of Target Data and Output Data on Static Eigenspace')
        ax.legend()
        self.plot_save_or_show(fig,"Residuals_Static")

    def plot_eigenval_pairs(self):
        """Plots the ensemble average NTK's eigenvalues vs time.  This is done in pairs so as to observe any eigenvalue crossings """
        assert (self.data.e_amount%2==0),"Uneven amount of eigenvalues" 
        idx = np.arange(self.data.e_amount)
        grouped = idx.reshape(-1,self.groupings)

        for i in range(grouped.shape[0]):
            fig, ax = plt.subplots(figsize = (10,6))
            for j in range(self.groupings):
                inst_idx = grouped[i,j]
                plt.plot(self.train_time_alignment,self.data.ensemble_evals[:,inst_idx], label = f"Eigenvalue {inst_idx+1}")
                # plt.fill_between(self.train_time_alignment, self.data.mean_evals[:,inst_idx] + self.data.mean_uncert_evals[:,inst_idx],
                                # self.data.mean_evals[:,inst_idx] - self.data.mean_uncert_evals[:,inst_idx], alpha = 0.3 )
            ax.set_xlabel(f"Training time")
            ax.set_ylabel("Eigenvalue Magnitude")
            ax.legend()
            ax.set_title("Eigenvalue vs Time")
            self.plot_save_or_show(fig, f"EvalPairs{i}")

    def plot_dout(self):
        """Plots the projection of the time derivative of the activation vector projected onto an eigenvector across time.  This is done in pairs of 2."""

        assert (self.data.e_amount%2==0),"Uneven amount of eigenvalues" 
        idx = np.arange(self.data.e_amount)
        grouped = idx.reshape(-1,self.groupings)

        for i in range(grouped.shape[0]):
            fig, ax = plt.subplots(figsize = (10,6))
            for j in range(self.groupings):
                inst_idx = grouped[i,j]
                # ax.plot(self.train_time_alignment, self.data.ens_resXeval[:,inst_idx], label = f"Eval {inst_idx+1} X Resid")
                ax.plot(self.train_time_alignment[:-1], self.data.ens_dout_proj[:,inst_idx], label = f"Proj of deriv on evec {inst_idx+1}")
            ax.set_xlabel(f"Training time")
            ax.set_ylabel("Magnitude")
            ax.legend()
            ax.set_title("Projection of activation rate onto eigenvector")
            self.plot_save_or_show(fig, f"DOutput{i}")

    def plot_reseval_dout(self):
        """Plots both the product of residuals and the corresponding eigenvalue, and the projection of the output vector onto the residual's eigenvector.
         NTK theory suggests these values should align in the infinite width limit.  There is a scale factor that is necessary here, since NTK theory is 
         based upon gradient flow, and we are using gradient descent.  
         
         The assert statement is simply to guarentee that eigenvalue pairs are plotted.  Since we use np.linalg.eigh to calculate eigenvalues/vectors at
         each timestep, there is a real possibility of eigenvalue crossing.  To be more explicit, if at time t0 lambda5>lambda6, then we rightly label the 
         eigenvalues as 5 and 6 respectively.  If at some later time, t' lambda6>lambda5, lambda6's value at t' will be internally considered to be the next 
         value of lambda5, thus introducing error.  To counter this, eigenvalues are typically plotted in pairs, so we can see in fine detail where any crossing
         may occur.  

           """
        assert (self.data.e_amount%2==0),"Uneven amount of eigenvalues" 
        idx = np.arange(self.data.e_amount)
        grouped = idx.reshape(-1,self.groupings)


        scale = (self.lr)/len(self.trial.x_train)

        for i in range(grouped.shape[0]):
            fig, ax = plt.subplots(figsize = (10,6))
            for j in range(self.groupings):
                inst_idx = grouped[i,j]
                ax.plot(self.train_time_alignment, scale*self.data.ens_resXeval[:,inst_idx],linewidth = 3,alpha = 0.6, label = f"Eval {inst_idx+1} X Resid")
                ax.plot(self.train_time_alignment[:-1], self.data.dout_proj[:,inst_idx],linewidth = 1.5, alpha = 0.9,
                        label = f"Proj of deriv on evec {inst_idx+1}")
            ax.set_xlabel(f"Training time")
            ax.set_ylabel("Magnitude")
            ax.legend()
            ax.set_title("Projection of activation rate onto eigenvector vs products of eigenvalues and residuals ")
            self.plot_save_or_show(fig, f"DOutput_VS_Resid{i}")

    def plot_performances_evals(self):
        """Plots the ensemble average performance (predicted output) and eigenvector components at various times specified by the performance times.
        Helpful to observe how and where the components of the individual eigenvectors change overtime.  It should be noted, that each eigenvector is
        subject to uncertainty via a product of -1, as there is no set/theoretical standard sign for each eigenvector.  That being said, the evolution 
        of the eigenvector is the important part, not sign."""
        x_vals = (self.trial.x_test.squeeze() * (4/3)*m.pi).numpy()
        x_train_vals = (self.trial.x_train.squeeze() * (4/3)*m.pi).numpy()

        idx = np.arange(self.data.e_amount)
        grouped_evals = idx.reshape(-1,self.groupings)

        ntk_epochs = np.array(self.trial.ntk_recorder.recorded_epochs)

        for i in range(grouped_evals.shape[0]):
            for k in range(self.data.ensemble_perf.shape[0]):
                fig, ax = plt.subplots(figsize = (10,6))
                epoch_value = self.performance_times[k]
                ax.plot(x_vals, self.data.ensemble_perf[k,:], label = f'Predicted Values')
                ax.plot(x_vals,self.trial.y_test.numpy(), label = f'True Values')
                # plt.scatter(self.x_train,self.y_train,label = "Training points")
                ax.scatter(x_train_vals, self.trial.y_train, label="Training points")
                ax.fill_between(x_vals, self.data.ensemble_perf[k,:]+self.data.ensemble_perf_uncert[k,:],
                                self.data.ensemble_perf[k,:]-self.data.ensemble_perf_uncert[k,:],color = 'blue', alpha = 0.3)
                #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()

                performance_epoch = self.trial.performance_recorder.recorded_epochs[k] #in terms of raw epochs not training time

                #Now adding eigenvectors
                ntk_time_idx = int(np.argmin(np.abs(ntk_epochs-performance_epoch))) #Grabs the lowest magnitude epoch which corresponds to ntk_recorded times - the time the performance is taken
                for j in range(self.groupings):
                    eval_idx =grouped_evals[i,j] 

                    # print(f'Length of self.data.ensemble_evecs[epoch,:,eval_idx] is {len(self.data.ensemble_evecs[int(epoch_value),:,eval_idx])}')
                    # print(f"length of x_vals is {len(x_train_vals)}")
                    ax.scatter(x_train_vals,self.data.ensemble_evecs[ntk_time_idx,:,eval_idx], marker='X', label =f'Eigenvector {eval_idx+1} component')
                ax.set_xlabel(f'X')
                ax.set_ylabel(f'Y')
                ax.set_title(f'Performance of the model at training time {epoch_value}')
                ax.legend()
                self.plot_save_or_show(fig,f"Performance{k+1}_WithEvec_{i}")
                
    def plot_relative_resXeval(self):
        """Plots the relative residuals multiplied by their corresponding eigenvalue vs time.  NTK theory would suggest that the change in activation rate 
        should correspond to the eigenvalue X their residual.  As such, plotting the product of the eigenvalue and their residual is a helpful sanity checker, 
        and way to touch base with the theory.  Plotting the relative version is helpful to observe which direction is the best to proceed in at any given time"""
        fig, ax = plt.subplots(figsize = (10,6))
        for k in range(self.data.e_amount):
            ax.plot(self.train_time_alignment, self.data.relative_resXeval[:,k], label = f"eval {k+1}") #kth vector component residual at all epochs
        self.add_NTK_points(ax)
        ax.set_xlabel('Training Time')
        ax.set_ylabel('Relative Size of Eigenvalue and Residual Product')
        ax.set_title(f'Relative size of Product of Residual and Eigenvalue vs Time')
        ax.legend()
        self.plot_save_or_show(fig, "RelativeResXEval")
    
    def plot_relative_res(self):
        """Plots the residuals vs time in relative residual space.  The relative residual space is normalised so we can see which residual is largest 
        excluding the eigenvalue multiplier at any given time"""
        fig, ax = plt.subplots(figsize = (10,6))
        for k in range(self.data.e_amount):
            ax.plot(self.train_time_alignment, self.data.relative_res[:,k], label = f"eval {k+1}") #kth vector component residual at all epochs
        self.add_NTK_points(ax)
        ax.set_xlabel('Training Time')
        ax.set_ylabel('Relative Size of Residual')
        ax.set_title(f'Relative size of Residual vs Time')
        ax.legend()
        self.plot_save_or_show(fig, "RelativeRes")
        
    def plot_ensemble_losses(self, uncertainty = True, plateau_points_post = False, t3_points = False, plateau_start_t3_points = False ):
        """Plots the losses on a replica by replica basis, as well as printing the overall average loss, and uncertainty.  
        
        Additionally, has the capability to plot:
        
        >Identified plateau start and end times from post-training processing.  At the time this code was written, the in loop tracker was not yet implemented
        and thus not used, however there is no particular reason to use the post training processing over in training processing.
        
        
        >t3 defined to be 1/(lambda3(t2)) where t2 = 1/(lambda2(t1)), and t1 is 1/(lambda1(0)).  The reason for this is we assume that NTK training works in 
        a sort of periscopic sense, and that training will initially follow the path (loosely) outlined by eigenvector 1 until time 1/lambda1(0), at which point
        the second eigenvalue is now the largest and most informative avenue of travel.  As such, a similar process occurs for eigenvalue 2, eigenvalue 3, 4 etc..
        requires DataProcessing's compute_timesOI
        
        >1/lambda3(t*) where t* is defined to be the time where plateau has started
        requires DataProcessing's compute_plateau_end_replicas

          """


        fig, ax = plt.subplots(figsize = (10,6))
        ax.plot(self.train_time_loss,self.data.ens_avg_loss, label = f"Ensemble Average")
        if uncertainty is True:
            ax.fill_between(self.train_time_loss, self.data.ens_avg_loss + self.data.ens_loss_uncert,
                            self.data.ens_avg_loss - self.data.ens_loss_uncert, alpha = 0.3)
        for i in range(self.trial.ensemble):
            ax.plot(self.train_time_loss,self.trial.loss_recorder.ensemble_loss[i,:],alpha = 0.5, linestyle = 'dashed')

            #Adding plateau_points from post training analysis
            if plateau_points_post is True:
                plateau_bounds = self.data.ens_plateau_bounds[i]
                for j in range(len(plateau_bounds)):
                    if plateau_bounds[j] is not np.nan:
                        ax.axvline(plateau_bounds[j]*self.lr, color = 'black', alpha = 0.3)

            #adding t3_times
            if t3_points is True:
                t3_times = self.data.times_OI[:,2] #Grab all replica's third time of interest
                if t3_times[i] is not np.nan:
                    ax.axvline(t3_times[i], alpha = 0.3, linestyle = 'dashed', color = 'black', label = 'Eval 3 at T2')


        #Adding eval3 calculated at plateau start
        if plateau_start_t3_points is True:
            for j in range(len(self.data.replica_plat_end_times)):
                ax.axvline(self.data.replica_plat_end_times[j], alpha = 0.3, linestyle = 'dashed', color = 'red', label = 'Eval3 at Plateau Start')
                

        #Making it so we only get one legend entry per item
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # keeps only the last handle per unique label
        ax.legend(by_label.values(), by_label.keys())


        ax.set_xlabel("Training Time") 
        ax.set_ylabel("Loss")
        ax.set_title("Replica and ensemble loss vs training time")
        self.plot_save_or_show(fig, "EnsembleAndReplica_Losses")

    def plot_replica_neurons(self):
        """Plot the neuron's pre-activations per replica on a layer by layer basis to see if there are any neurons which are particularly active"""
        hidden_layers =self.data.numlayer-1
        for i in range(hidden_layers):
            phi_layer = self.trial.phi[i]
            for j in range(self.data.ensemble):
                fig, ax = plt.subplots(figsize = (10,6))
                for k in range(self.trial.width):
                    ax.plot(self.train_time_loss, phi_layer[j,:,k]) 
                    #Grab the jth replica's ith layer pre-activations across all epochs for neuron k
                ax.set_xlabel("training time")
                ax.set_ylabel("Pre-activation magnitude")
                ax.set_title("Pre-activation of neuron vs training time")
                self.plot_save_or_show(fig, f"Replica_{j}_Layer{i}")

    def plot_weight_deriv(self):
        first_layer = self.data.ens_weight_deriv[0]
        for i in range(first_layer.shape[0]): #Replicas
            replica = first_layer[i] #Now 0th axis is epochs
            fig, ax = plt.subplots(figsize = (10,6))
            # print(f'Replica shape is {replica.shape}')
            for j in range(replica.shape[1]):
                ax.plot(self.train_time_alignment[:-1],replica[:,j,:])
            self.add_NTK_points(ax)
            ax.set_xlabel("Training Time")
            ax.set_ylabel("Derivative Magnitude")
            ax.set_title(f"Derivative of Weights Vs Training Time For First Hidden Layer in Replica {i}")
            self.plot_save_or_show(fig, f"Replica_{i}_Layer1WeightDeriv")

    def plot_bias_deriv(self):
        for k in range(len(self.data.ens_bias_deriv)):
            layer = self.data.ens_bias_deriv[k]
            for i in range(layer.shape[0]): #Replicas
                replica = layer[i] #Now 0th axis is epochs
                fig, ax = plt.subplots(figsize = (10,6))
                # print(f'Replica shape is {replica.shape}')
                for j in range(replica.shape[1]):
                    ax.plot(self.train_time_alignment[:-1],replica[:,j])
                ax.set_xlabel("Training Time")
                ax.set_ylabel("Derivative Magnitude")
                ax.set_title(f"Derivative of Biases Vs Training Time For Hidden Layer {k+1} in Replica {i}")
                self.add_NTK_points(ax)
                self.plot_save_or_show(fig, f"Replica_{i}_Layer_{k+1}_BiasDeriv")

    def plot_weight_derivs_hist(self):#Currently only works for alignmentint1
        """Create a printout of the derivative of the weight tensor on a replica by replica basis.  This is meant primarily to focus on specific regions of interest
        such as the initial loss drop.  Using this for large t experiments is not recommended."""
        times = self.lr *self.data.dw_times
        for i in range(1,len(self.data.ens_bias_deriv)): #Don't need for layer 1, that is a vector of size 10
            layer_derivs = self.data.delta_weight_dict[i] #remaining dim [replicas,snapshots-1,10,10]
            for j in range(self.trial.ensemble): #Loop over ensemble members
                replica_delta = layer_derivs[j,:]
                for k in range(replica_delta.shape[0]):
                    flattened_array = replica_delta[k,:,:].ravel() #Ravel preferred here for memory conservation
                    bins= int(len(np.unique(flattened_array))/2) #5 is a test
                    fig,ax = plt.subplots(figsize = (10,6))
                    ax.hist(flattened_array, bins = bins)
                    ax.set_xlabel(f"$\Delta$ W")
                    ax.set_ylabel("Counts")
                    ax.set_title(f"Change in Weight Distribution for Replica {j}, Layer{i+1}, from t {times[k]} to {times[k+1]}")
                    self.plot_save_or_show(fig,f"Replica_{j}_Layer_{i+1}_DeltaW")

    def plot_weight_derivs_heatmap(self):
        times = self.lr *self.data.dw_times
        for i in range(1,len(self.data.ens_bias_deriv)): #Don't need for layer 1, that is a vector of size 10
            layer_derivs = self.data.delta_weight_dict[i] #remaining dim [replicas,snapshots-1,10,10]
            for j in range(self.trial.ensemble): #Loop over ensemble members
                replica_delta = layer_derivs[j,:]
                for k in range(replica_delta.shape[0]):
                    replica_delta_time = replica_delta[k,:,:]
                    vmin = np.min(replica_delta_time)
                    vmax = np.max(replica_delta_time)
                    fig, ax = plt.subplots(figsize =(10,6))
                    sns.heatmap(replica_delta_time,vmin =vmin, vmax =vmax, cmap = 'coolwarm', cbar_kws={'label': 'Change in Weights'}, ax =ax)    
                    ax.set_xlabel(f"Layer {i} neuron output")
                    ax.set_ylabel(f"Layer {i+1} neuron")
                    ax.set_title(f"Change in Weight Distribution for Replica {j}, Layer{i+1}, from t {times[k]} to {times[k+1]}")
                    self.plot_save_or_show(fig, f"Replica_{j}_Layer_{i+1}_DeltaW_Heatmap")
                    
    def plot_weight_derivs_hist_plat(self):

        for i in range(1,len(self.data.ens_bias_deriv)): 
            fig, ax = plt.subplots(figsize = (10,6))
            bins = int(len(np.unique(self.data.ens_dweight_hist_vals[i]))/10) 
            ax.hist(self.data.ens_dweight_hist_vals[i], bins = bins)
            ax.set_xlabel(f"$\Delta$ W")
            ax.set_ylabel("Counts")
            ax.set_title(f"Change in weight distribution in layer {i+1} from Plateau Start to Plateau End")

            self.plot_save_or_show(fig, f"Layer_{i+1}_DeltaW_Plat")
            
    #Come back to this, add a conditional statement that saves the slopes from the data proccessing function and just plots that as opposed to doing the whole thing again
    def plot_slopes(self):

        window = 0.5 #training time
        num_epochs = max(2,int(round(window/self.trial.lr))) #From 0.5 window -> 50 epochs

        epochs_axis = np.arange(self.trial.loss_recorder.ensemble_loss.shape[1]) #Grabs the number of epochs in the trial.loss_recorder.ensemble_loss   
        x = np.array(self.trial.loss_recorder.recorded_epochs) *self.trial.lr

        x_window = x[:num_epochs] #Just the length of a window
        x_centered = x_window - x_window.mean()
        denom = np.sum(x_centered**2) #Denominator for least squares regression is sum over i (xi - xbar)^2, this will be the same for all our windows
        #Want a least squares slope to take in all the data of the window, not just the endpoints which can be noisy
        kernel = x_centered[::-1] #need to flip it now since convolution will flip it again,

        for i in range(self.data.ensemble):

            plateau_bounds = self.data.ens_plateau_bounds[i]

            inst_loss = self.trial.loss_recorder.ensemble_loss[i,:] #replica i's information across all epochs
            log_loss = np.log(np.clip(inst_loss,1e-12,None)) #Ensure that any value below 1e-12 (i.e.,) 0 are filtered out, to avoid a crash      

            slopes = np.convolve(log_loss,kernel, mode = 'valid')/denom #mode = valid ensures we only have a scalar, the individual slope for that selected window
            """Dimensionality of slopes:
            slopes[k] would retreive the slope of the window covering epochs k through k+num_epochs-1
            """
            slopes_epochs = epochs_axis[:len(slopes)] #since slopes[k] covers [k,k+num_epochs-1] 

            #Now converting it to a log of loss
            fig, ax = plt.subplots(figsize = (10,6))
            ax.plot(x[slopes_epochs], slopes)
            for j in range(len(plateau_bounds)):
                if plateau_bounds[j] is not np.nan:
                    ax.axvline(plateau_bounds[j]*self.lr, color = 'black', alpha = 0.3)
            ax.set_ylim(-.5,0.5)
            ax.set_xlabel("Training time")
            ax.set_ylabel("D(ln(loss))/dt")
            ax.set_title("Log Loss vs Training Time")
            if self.savefig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\Replica_{i}_Slopes.png')
            if self.show:
                plt.show()
            plt.close()            

    def plot_replica_losses_inlooptrack(self): 
        """Plots the ensemble and replica losses, with in-loop-tracked plateau start and end points marked with black lines"""
        plateau_data = self.trial.plateau_recorder.plateau_bounds #dim is [replica, 2]
        fig, ax = plt.subplots(figsize = (10,6))
        ax.plot(self.train_time_loss,self.data.ensemble_average_loss, label = f"Ensemble Average")
        for i in range(self.trial.ensemble):
            # inst_loss = self.trial.loss_recorder.ensemble_loss[i]
            ax.plot(self.train_time_loss,self.trial.loss_recorder.ensemble_loss[i,:],alpha = 0.5, linestyle = 'dashed')
            # converted_point = self.data.replica_plat_end_times[i]
            
            #Adding points
            plateau_bounds = plateau_data[i]
            for j in range(len(plateau_bounds)):
                if plateau_bounds[j] is not np.nan:
                    ax.axvline(plateau_bounds[j]*self.lr, color = 'black', alpha = 0.3)
    
        ax.set_xlabel("Training Time") 
        ax.set_ylabel("Loss")
        ax.set_title("Replica and ensemble loss vs training time")
        ax.legend()
        self.plot_save_or_show(fig, "ReplicaLosses_InLoop")  

    def plot_eval3_hist(self):
        """Plots histogram of the range of eigenvalue 3 values at both plateau start (as identified by the plateau_recorder in the trial file),
        and at t2 defined to be t2 = 1/lambda2(t1) where t1 = 1/lambda1(0)"""

        plateau_data = self.data.plat_end_eval3

        t2_data = self.data.eval3_at_t2 

        bins_identifier = plateau_data+ t2_data

        num_unique = len(np.unique(np.array(bins_identifier)))

        # num_bins = int(num_unique/5)
        # print(f"Number of bins is {num_bins}")

        num_bins = 40

        fig,ax = plt.subplots(figsize = (10,6))
        ax.hist(plateau_data, bins = num_bins, label = 'Eval3 at Plateau Start', alpha = 0.6)
        ax.hist(t2_data, bins = num_bins, label = 'Eval3 at T2', alpha = 0.6)
        ax.set_xlabel(f"$\lambda$3 ")
        ax.set_ylabel("Counts")
        ax.set_title(f"Distribution of eval 3")
        ax.legend()
        self.plot_save_or_show(fig, "Eval3Hist")

    def platend_eval3_barchart(self):
        """Produce a bar chart of the inverse eval3 values compared with the observed time of plateau end"""
        fig, ax = plt.subplots(figsize = (10,6))
        x_iter = 0 #value for x position on the bar chart, iterate so that when we remove replicas that did not get out of the plateau, the spacing is not destroyed
        inst_list = [] #list to store all the values we are interested in plotting, i.e., plateau end and inverse eval3 at various times
        labels = ["Plateau End", "Eval3 inverse at t2", "Eval3 inverse at plat start"]
        colors = ["red", "green", "blue"]
        alphas = [0.8, 0.5,0.5] #Real plateau end will occur first (almost always), thus we want that to be more apparent

        rgba_colors = [mcolors.to_rgba(c, alpha=a) for c, a in zip(colors, alphas)] #Have to bake the alpha into the colors, cannot pass a list of values to the alpha

        bar_width = 0.25
        for i in range(self.trial.ensemble):
            if np.isnan(self.trial.plateau_recorder.plateau_bounds[i,1]):
                #check if the ith replicas plateau end is a np.nan
                pass
            else:
                #if a plateau end was recorded, we do want to plot these values
                inst_list.append(self.trial.plateau_recorder.plateau_bounds[i,1] *self.lr) #recall that the recorder only tracks epochs, need to convert to train time
                inst_list.append(self.data.times_OI_unfiltered[i,2]) #times_OI contain 3 points per replica, see docstring at top of function in DataProcessing
                inst_list.append(self.data.replica_plat_end_times_unfiltered[i]) #replica plat end times does not contain more than 1 point per replica

                x_pos = [x_iter +j * bar_width for j in range(3)]

                # ax.bar(x_iter,inst_list,width = bar_width, alpha = alphas, label = labels, color = colors, edgecolor = "black", linewidth = 0.75)
                ax.bar(x_pos,inst_list,width = bar_width, label = labels, color = rgba_colors, edgecolor = "black", linewidth = 0.75)

                x_iter +=1 #iterate the counter
                inst_list = [] #reset the list

        ax.set_xlabel("Replica")
        ax.set_ylabel("Training Time")
        ax.set_title("Recorded Plateau End vs Eval 3 Inverse")
        
        #Making it so we only get one legend entry per item
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # keeps only the last handle per unique label
        ax.legend(by_label.values(), by_label.keys())

        self.plot_save_or_show(fig,"PlatEndBar")

    def plot_initial_evals_barchart(self):
        """Plots initial eigenvalues on a per replica basis to assure that our results conform with what theory suggests; that at finite
        width, all eigenvalues should be approximately the same
        
        This is done using the eigenvalues calculated from DataProcessing's compute_eigen_quantity, that is to say it is a per-replica result,
        not averaged over the whole, and with no explicit uncertainty
        """

        for i in range(self.data.e_amount):
            ensemble_eval = self.trial.ntk_recorder.initial_evals[:,i]
            fig, ax = plt.subplots(figsize = (10,6))
            for j in range(self.trial.ensemble):
                ax.bar(j,ensemble_eval[j], width = 0.50, edgecolor = "black", linewidth = 0.75)
            ax.set_xlabel("Replica")
            ax.set_ylabel(f"Eigenvalue {i+1}")
            ax.set_title(F"Eigenvalue {i+1} at Initialisation For Each Replica")
            self.plot_save_or_show(fig, f"Eval{i+1}_InitBar")

    def plot_initial_evals_ensemble_barchart(self):
        """Plots the ensemble average initial eigenvalues, taken using DataProcessing's compute_eigvals_zero
        This contains not only the ensemble average NTK at initilisation's eigenvalues, but also the bootstrapped uncertainty over the 
        spectrum of bootstrapped NTK's eigenvalues.

        In order to observe whether any replica is particularly varying from the norm, this plot is implemented in a similar way to the initial_evals (i.e., relica 
        barchart), however the ensemble average is also plotted as a series of horizontal lines (thus forming a region), allowing one to easily see where true variation
        from expected value is occuring.

        required DataProcessing functions:
        compute_eigvals_zero
        compute_eigen_quantity

        """

        for i in range(self.data.e_amount):
            average_eval = self.data.mean_eigenvals[i]
            average_eval_uncert = self.data.mean_eigenvals_uncert[i]
            replicas_eval = self.trial.ntk_recorder.initial_evals[:,i]
            fig, ax = plt.subplots(figsize = (10,6))
            for j in range(self.trial.ensemble):
                ax.bar(j,replicas_eval[j], edgecolor = "black", linewidth = 0.75)

            #Adding horizontal sections
            ax.axhline(average_eval,color = 'black', linestyle = '--', linewidth = 1.5, label = "Ensemble Average")
            ax.axhspan(average_eval-average_eval_uncert, average_eval+average_eval_uncert,color = 'black', alpha = 0.3)

            ax.set_xlabel("Replica")
            ax.set_ylabel(f"Eigenvalue {i+1}")
            ax.set_title(F"Eigenvalue {i+1} at Initialisation For Each Replica")
            ax.legend()
            self.plot_save_or_show(fig,f"Eval{i+1}_EnsembleAVGBar")

    def plot_eval3_t2_barchart(self):
        """Creates a bar chart of all replica's and their corresponding eval3 at time t2.  This is done in order to see if there is in fact
        any correspondence to plateau end time
        
        requires: calculate_timesOI, and compute_ens_eval3_t2

        """

        fig, ax = plt.subplots(figsize = (10,6))
        for i in range(self.trial.ensemble):
            ax.bar(i, self.data.eval3_at_t2[i], linewidth = 0.75, edgecolor = 'black')
        ax.axhline(self.data.filtered_eval3_mean, color = 'black', linestyle = 'dashed', linewidth = 2, label = "Filtered Average")
        ax.axhline(self.data.eval_3_mean, color = 'blue', linestyle = 'dotted', linewidth = 2, label = "Unfiltered Average")
        ax.set_xlabel("Replica")
        ax.set_ylabel(f"$\lambda_3$")
        ax.set_title(f"$\lambda_3$ at t2 distribution")
        ax.legend()
        self.plot_save_or_show(fig,"Eval3_at_t2")

    def plot_plat_end_t3_barchart(self):
        """Creates a barchart for all replica's plateau end time, also plots both the filtered and unfiltered ensemble average t3
        requires: calculate_timesOI and compute_ens_eval3_t2
        """
        fig, ax = plt.subplots(figsize = (10,6))
        for i in range(self.trial.ensemble):
            if np.isnan(self.trial.plateau_recorder.plateau_bounds[i,1]):
                #check if the ith replicas plateau end is a np.nan
                pass
            else:
                #if a plateau end was recorded, we do want to plot these values
                ax.bar(i, (self.trial.plateau_recorder.plateau_bounds[i,1]*self.lr), linewidth = 0.75, edgecolor = 'black')

        #sanity checker
        print(self.data.filtered_t3_mean)
        print(self.data.t3_mean)


        ax.axhline(self.data.filtered_t3_mean, color = 'black', linestyle = 'dashed',linewidth = 2, label = "Filtered Average")
        ax.axhline(self.data.t3_mean, color = 'blue', linestyle = 'dotted', linewidth = 2,label = "Unfiltered Average")
        ax.set_xlabel("Replica")
        ax.set_ylabel(f"Plateau End Time")
        ax.set_title(f"Plateau End Time Distribution")
        ax.legend()
        self.plot_save_or_show(fig,"PlateauEndT3Bar")

    def plot_plat_start_t2_barchart(self):
        """Plots the replica's recorded plateau start alongside their recorded t2 times, to see whether there is correlation 
        Requires compute_timesOI
        """
        bar_width = 0.4
        fig, ax = plt.subplots(figsize = (10,6))
        for i in range(self.trial.ensemble):
            ax.bar(i, self.data.times_OI_unfiltered[i,1],width=bar_width, linewidth = 0.75, edgecolor = 'black', color = 'blue',
                label = 't2')
            ax.bar((i+bar_width), self.trial.plateau_recorder.plateau_bounds[i,0]*self.lr,width = bar_width,linewidth = 0.75,
                    edgecolor = 'black', color ='orange', label = 'recorded start' )
        ax.set_xlabel("Replica")
        ax.set_ylabel(f"Training Time")
        ax.set_title(f"Plateau Start Time Distribution")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # keeps only the last handle per unique label
        ax.legend(by_label.values(), by_label.keys())        
        self.plot_save_or_show(fig,"t2_platstart_barchart")

    def plot_evec3_plateau_rotation(self,target_eval = 3):
        """Creates a plot of the ensemble average eigenvector 3 rotation (as measured from evec3(t2)) during the plateau, as well
        as replica printouts of the same quantity for replicas which did manage to leave the plateau
        requires:
        Data.compute_eigen_quant
        Data.compute_timesOI 
        Data.compute_ensemble_evec3_plateau_rotate
        Data.compute_replica_plateau_evec_rotations"""

        flat_replica_times = np.concatenate(list(self.data.replica_plateau_rotations_times.values()))

        replica_xmin = np.min(flat_replica_times)
        ens_xmin = np.min(np.array(self.data.plateau_ntk_times))

        replica_xmax = np.max(flat_replica_times)
        ens_xmax = np.max(np.array(self.data.plateau_ntk_times))

        plot_xmin = min(int(replica_xmin),int(ens_xmin))
        plot_xmax = max(int(replica_xmax),int(ens_xmax))
        
        ensemble_time =np.array(self.data.plateau_ntk_times)*self.lr 
        fig, ax = plt.subplots(figsize = (10,6))
        ax.plot(ensemble_time, self.data.ens_plateau_evec3_rotations, 
                label= "Ensemble Average", color = 'blue', alpha = 0.8)
        for key in self.data.replica_plateau_rotations.keys():
            x_axis =np.array(self.data.replica_plateau_rotations_times[key]) *self.lr 
            ax.plot(x_axis, self.data.replica_plateau_rotations[key],
                    linestyle = 'dashed', alpha = 0.5, label = f"Replica {key}")
        ax.set_xlim(plot_xmin*self.lr,plot_xmax*self.lr)
        ax.set_ylabel(f"Eigenvector {target_eval} Rotation in Radians")
        ax.set_xlabel("Training Time")
        ax.set_title(f"Eigenvector {target_eval} Rotation in Plateau Relative to Plateau Start")
        ax.legend()
        self.plot_save_or_show(fig, f"evec{target_eval}_plateau_rotation")


    def plot_evec_overlap_TOI(self, num_times =3, num_evals =3, sorted =True):
        """Plots a heatmap of eigenvector overlap for each replica on a per replica basis.  The time of selection of replica eigenvector is each replica's 
         individual t1, t2 etc. 
         
         Note to Logan, should add a way to calculate num_evals/num_times off the passed arrays and not just inputting them) 
         """

        convergence_list = np.array(self.trial.loss_recorder.convergence_tracker)
        if sorted:
            indices = self.data.replica_sort()
            convergence_list = convergence_list[indices]

        for i in range(num_evals):
            evec_array = self.data.evec_overlap_TOI[i]

            for j in range(num_times):
                overlap_array = evec_array[j,:]
                min_val = np.nanmin(overlap_array)
                max_val = np.nanmax(overlap_array)
                fig, ax = plt.subplots(figsize = (10,6))
                sns.heatmap(overlap_array,vmin =min_val, vmax =max_val, cmap = 'coolwarm', cbar_kws={'label': 'Eigenvector Alignment'}, ax =ax)

                labels = ax.get_xticklabels()
                ylabels = ax.get_yticklabels()
                for idx, label in enumerate(labels):
                    if convergence_list[idx]:
                        label.set_fontweight('bold')
                        ylabels[idx].set_fontweight('bold')

                ax.collections[0].cmap.set_bad('grey')
                #Set the colormap for any 'bad values' (i.e., any np.nans) as grey

                ax.set_xlabel("Replica Number")
                ax.set_ylabel("Replica Number")
                ax.set_title(f"Alignment of Replica Evec {i+1}, at corresponding time t{j+1}")
                self.plot_save_or_show(fig, f"Replica_TOI{j+1}_Evec{i+1}_Align") 
                   
    def plot_ensembleAVG_evec_overlap_TOI(self, num_times =2, num_evals =3, sorted = True):
        """Plots a heatmap of eigenvector overlap for each replica on a per replica basis.  The time of selection of replica eigenvector is each replica's 
         individual t1, t2 etc. 
         
         """

        convergence_list = np.array(self.trial.loss_recorder.convergence_tracker)
        if sorted:
            indices = self.data.replica_sort()
            convergence_list = convergence_list[indices]



        for i in range(num_evals):
            evec_array = self.data.ensembleAVG_evec_overlap_TOI[i]
            for j in range(num_times):
                overlap_array = evec_array[j,:]
                min_val = np.nanmin(overlap_array)
                max_val = np.nanmax(overlap_array)
                fig, ax = plt.subplots(figsize = (10,6))
                sns.heatmap(overlap_array,vmin =min_val, vmax =max_val, cmap = 'coolwarm', cbar_kws={'label': 'Eigenvector Alignment'}, ax =ax)

                ylabels = ax.get_yticklabels()
                labels = ax.get_xticklabels()
                for idx, label in enumerate(labels):
                    if convergence_list[idx]:
                        label.set_fontweight('bold')
                        ylabels[idx].set_fontweight('bold')

                ax.collections[0].cmap.set_bad('grey')
                #Set the colormap for any 'bad values' (i.e., any np.nans) as grey

                ax.set_xlabel("Replica Number")
                ax.set_ylabel("Replica Number")
                ax.set_title(f"Alignment of Replica Evec {i+1}, at ensemble NTK time t{j+1}")
                self.plot_save_or_show(fig, f"ensembleAVG_Replica_TOI{j+1}_Evec{i+1}_Align") 
                   


    """Below is a set of functions meant to create a visual of the network, not particularly useful for data analysis, 
    but helpful for any formal writing/presentation"""
    def build_visual_nx(self):
        """Code is meant to be a framework for building a visual of the network using NetworkX package"""

        G = nx.DiGraph()

        node_id = 0
        """This is an iterative counter, will be important for knowing a nodes 'absolute position'"""
        effective_width = self.trial.model.layer_widths[:-1]
        layer_start_ids = [] #Stores the layer ids

        for layer_idx, widths in enumerate(effective_width):
            layer_start_ids.append(node_id) #append the layer id (i.e., 1-4)
            for neuron_idx in range(widths): #For all the neurons in the layer
                G.add_node(node_id,layer = layer_idx, neuron_idx = neuron_idx) #Attaching labels to this specificn neuron, will be helpful later on
                node_id+=1

                """node_id at the end will be of shape [0,10,20,30]"""
        """We have now fully populated the nodes required to visualise the network, now add the edges (connecting lines)"""

        for layer_idx in range(len(effective_width)-1): #Only want consecutive pairs of layers, hence the -1
            current_layer_start = layer_start_ids[layer_idx] #grabs starting indi
            next_layer_start = layer_start_ids[layer_idx+1]
            #Above does what says, grabs the current and next layer
            for i in range(effective_width[layer_idx]): #For number of neurons in current layer
                for j in range(effective_width[layer_idx+1]):
                    G.add_edge(current_layer_start + i, next_layer_start +j) #adds the connection 

        return G, layer_start_ids
    
    def nodes_positions(self,G):
        """Need to manually set the positions of the neurons so it stays consistent across all snapshots
        x = layer index, y corresponds to the position of neuron within its layer
        """
        positions = {} #Create empty dictionary to store all the positions
        node_id = 0 

        effective_width = self.trial.model.layer_widths[:-1]
        for layer_idx, widths in enumerate(effective_width):
            for neuron_idx in range(widths):
                y = neuron_idx - (widths-1)/2
                positions[node_id] = (layer_idx, y)
                node_id+=1
        return positions
    
    def make_snapshot(self,G,positions,nodesize,edgesize):

        colormap =plt.cm.viridis


        """Need to make self.data.phi_neuron into a more readable format for vmin and vmax"""
        neuron_flat = np.concatenate([self.data.phi_neuron[k].flatten() 
                                      for k in range(len(self.trial.model.layer_widths)-1)])
        #Added the -1 to exclude the final neuron, which will be huge 
        


        c_min = np.min(neuron_flat)
        c_max = np.max(neuron_flat)
        norm = matplotlib.colors.Normalize(vmin = c_min, vmax =c_max)

        effective_width = self.trial.model.layer_widths[:-1]
        
        """Same structure as the neurons themselves for the midpoints"""
        t_iter= 0 
        for (t_start, t_end) in self.regions:
            fig,ax = plt.subplots(figsize=(10,10))
            time_of_int =t_start/2 + t_end/2 #Grabs the midpoint of the region
            epoch_id = int(time_of_int/self.lr)-self.trial.record_start #Converts it back into the epochs


            """Grabs the neurons values at the corresponding time (epoch_id)"""
            magnitudes = []
            for layer_idx, widths in enumerate(effective_width):
                values = self.data.phi_neuron[layer_idx][epoch_id,:] 
                magnitudes.extend(values)


            nx.draw_networkx_nodes(G,positions,ax =ax, node_color = magnitudes,cmap = colormap, node_size=nodesize,
                                 vmax = c_max, vmin = c_min)
            nx.draw_networkx_edges(G,positions, ax = ax, width = edgesize, alpha = 0.4)
            sm = plt.cm.ScalarMappable(norm = norm, cmap =colormap)
            fig.colorbar(sm, ax =ax, label = 'Magnitude',fraction = 0.05) #Fraction controls how much of the fig is allocated to the colorbar
            ax.set_title(f"Network at time {time_of_int}")
            ax.axis("off") #Turns off axes

            self.plot_save_or_show(fig, f"ModelSnapshot_{t_iter}_{t_iter+1}")

            t_iter+=1
    
    def plot_model_snapshots_midpoints(self):
        G, layer_start_ids = self.build_visual_nx()
        positions = self.nodes_positions(G)

        self.make_snapshot(G,positions,self.nodesize,self.edgesize)


    """Below is a (outdated) set of functions meant to produce radial plots of preactivations.  Plotting radially allows us to see all the neurons
    in a layer at one time, which is rather helpful in a non-trivial network.  These were used intermittently in my experimentation, although very inconsistently;
    they have not been kept up to date"""
    def plot_neuron_preactivations_regions(self):
        cmap = plt.cm.tab10

        theta_threshold = np.linspace(0,2*m.pi,150) #Smooth amount of radians
        t_iter=0
        for (t_start, t_end) in self.regions:
            time_of_int =t_start/2 + t_end/2 #Grabs the midpoint of the region
            epoch_id = int(time_of_int/self.lr)-self.trial.record_start #Converts it back into the epochs

            self.N = sum(self.trial.model.layer_widths) #Sums the model's widths so we know how many neurons we have
            angles = np.linspace(0, 2*m.pi, self.N, endpoint= False) #Creates self.N linearly spaced points between 0 and 2pi
            #Need endpoint = false to prevent overlap of points on 0 / 2pi
 
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111,projection= 'polar')
            

            magnitudes = []
            layer_ids = []

            start = 0

            for layer_idx, widths in enumerate(self.trial.model.layer_widths):
                values = self.data.phi_neuron[layer_idx][epoch_id,:] 
                """Grabs the layer id element of the self.phi_neurons dictionary, 
                and then grabs all the neuron's activations corresponding to the particular time of interest"""
                magnitudes.extend(values) #Adds all 10 (or 1) values to the magnitudes list
                layer_ids.extend([layer_idx]*widths) #Correctly populates the layer_ids list with the necessary labels

            #Plotting actual points
            for layer_idx, widths in enumerate(self.trial.model.layer_widths):
                layer_angles = angles[start:start+widths]
                layer_mags = magnitudes[start:start+widths]
                ax.scatter(layer_angles, layer_mags, color=cmap(layer_idx / max(len(self.trial.model.layer_widths)-1, 1)),
                s=40, label=f'Layer {layer_idx}')
                start += widths
            # ax.scatter(angles, magnitudes, c=colors, s=40) #s just corresponds to the points size

            #Plotting phi threshold
            ax.plot(theta_threshold,np.full_like(theta_threshold,self.data.p_target), linestyle = 'dotted', color = 'black', alpha = 0.6)
            ax.set_xticks([]) #removes angle markers
            ax.set_title(f"Pre-activation of neurons at time {time_of_int}")
            ax.legend()
            if self.savefig:
                fig.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\RadialPhi_{t_iter}_{t_iter+1}.png')
            plt.close(fig)
            t_iter+=1

    def plot_neuron_preactivations_reduced_regions(self):
        cmap = plt.cm.tab10

        theta_threshold = np.linspace(0,2*m.pi,150) #Smooth amount of radians
        t_iter=0
        for (t_start, t_end) in self.regions:
            time_of_int =t_start/2 + t_end/2 #Grabs the midpoint of the region
            epoch_id = int(time_of_int/self.lr)-self.trial.record_start #Converts it back into the epochs

            self.N = sum(self.trial.model.layer_widths) #Sums the model's widths so we know how many neurons we have
            angles = np.linspace(0, 2*m.pi, self.N, endpoint= False) #Creates self.N linearly spaced points between 0 and 2pi
            #Need endpoint = false to prevent overlap of points on 0 / 2pi
 
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111,projection= 'polar')
            

            magnitudes = []
            layer_ids = []

            start = 0

            reduced_widths = self.trial.model.layer_widths[:-1] #removes the last entry, i.e., the 1 neuron

            for layer_idx, widths in enumerate(reduced_widths):
                values = self.data.phi_neuron[layer_idx][epoch_id,:] 
                """Grabs the layer id element of the self.phi_neurons dictionary, 
                and then grabs all the neuron's activations corresponding to the particular time of interest"""
                magnitudes.extend(values) #Adds all 10 (or 1) values to the magnitudes list
                layer_ids.extend([layer_idx]*widths) #Correctly populates the layer_ids list with the necessary labels

            #Plotting actual points
            for layer_idx, widths in enumerate(reduced_widths):
                layer_angles = angles[start:start+widths]
                layer_mags = magnitudes[start:start+widths]
                ax.scatter(layer_angles, layer_mags, color=cmap(layer_idx / max(len(self.trial.model.layer_widths)-1, 1)),
                s=40, label=f'Layer {layer_idx}')
                start += widths
            # ax.scatter(angles, magnitudes, c=colors, s=40) #s just corresponds to the points size

            #Plotting phi threshold
            ax.plot(theta_threshold,np.full_like(theta_threshold,self.data.p_target), linestyle = 'dotted', color = 'black', alpha = 0.6)
            ax.set_xticks([]) #removes angle markers
            ax.set_title(f"Pre-activation of neurons at time {time_of_int}")
            ax.legend()
            if self.savefig:
                fig.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\RadialPhi_{t_iter}_{t_iter+1}.png')
            plt.close(fig)
            t_iter+=1

    def plot_neuron_preactivations(self,time):
        cmap = plt.cm.tab10

        theta_threshold = np.linspace(0,2*m.pi,150) #Smooth amount of radians

        time_of_int =time 
        epoch_id = int(time_of_int/self.trial.lr) #Converts it back into the epochs

        self.N = sum(self.trial.model.layer_widths) #Sums the model's widths so we know how many neurons we have
        angles = np.linspace(0, 2*m.pi, self.N, endpoint= False) #Creates self.N linearly spaced points between 0 and 2pi
        #Need endpoint = false to prevent overlap of points on 0 / 2pi

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111,projection= 'polar')
        start = 0
        magnitudes = []
        layer_ids = []
        for layer_idx, widths in enumerate(self.trial.model.layer_widths):
            values = self.data.phi_neuron[layer_idx][epoch_id,:] 
            """Grabs the layer id element of the self.phi_neurons dictionary, 
            and then grabs all the neuron's activations corresponding to the particular time of interest"""
            magnitudes.extend(values) #Adds all 10 (or 1) values to the magnitudes list
            layer_ids.extend([layer_idx]*widths) #Correctly populates the layer_ids list with the necessary labels
        
        #Plotting actual points
        # ax.scatter(angles, magnitudes, c=colors, s=40) #s just corresponds to the points size
        for layer_idx, widths in enumerate(self.trial.model.layer_widths):
            layer_angles = angles[start:start+widths]
            layer_mags = magnitudes[start:start+widths]
            ax.scatter(layer_angles, layer_mags, color=cmap(layer_idx / max(len(self.trial.model.layer_widths)-1, 1)),
            s=40, label=f'Layer {layer_idx}')
            start += widths
        #Plotting phi threshold
        ax.plot(theta_threshold,np.full_like(theta_threshold,self.data.p_target), linestyle = 'dotted', color = 'black', alpha = 0.6)
        ax.set_title(f"Pre-activation of neurons at time {time_of_int}")
        ax.set_xticks([]) #removes angle markers

        ax.legend()
        if self.savefig:
            fig.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.filename}\RadialPhi_{time_of_int}.png')
        plt.close(fig)
