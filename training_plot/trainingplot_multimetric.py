import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import keras
import numpy as np


class TrainingPlot_Multimetric(keras.callbacks.Callback):
 
    def __init__(self, filename = './output2/training_plot.jpg', metrics=[]):
        
        # The metrics variable expects a list of metrics such as accuracy, precision, recall          
        self.filename = filename
        # hist dictionary will store all the metrics to be plotted.
        # in the form of {"loss" : [list of losses], "val_loss": [list of val_losses] ...}
        self.hist = {}
        self.metrics = metrics
        
        
        # initializing the metrics lists, every metric has training and validation part
        self.hist['loss'] = []
        self.hist['val_loss'] = []         
        if len(self.metrics) != 0:
            for metric in self.metrics:
                self.hist[metric] = []
                self.hist[f"val_{metric}"] = []
    
    # this function will be called when the training begins
    def on_train_begin(self, logs={}):
        self.hist['logs'] = []
    
    # this function will be called at the end of every epoch.
    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss and metrics for the entire training process
        self.hist['logs'].append(logs)
        self.hist['loss'].append(logs.get('loss'))
        #print(f"\n loss - {len(self.hist['loss'])}")
        self.hist['val_loss'].append(logs.get('val_loss'))
      
      
        # checking if metrics list is supplied
        if len(self.metrics) != 0:
            # looping over keys in the logs
            for key in logs.keys():

                # Since loss list is already updated, we shouldn't include them
                if key!="loss" and key!="val_loss":

                    # Sometimes the keys in the logs end with a number for eg precision_3, recall_12
                    # This is not the case with the accuracy and val_accuracy.
                    if not key[-1].isdigit():  
                        self.hist[key].append(logs.get(key))

                    # if the key ends with a number, we compare the supplied metric with the starting part of the key.
                    # if the key matches, then we have found the key.  
                    else:
                        to_app = ""
                        for prop in self.hist.keys():
                            if prop == key[:len(prop)]:
                                to_app = prop
                        self.hist[to_app].append(logs.get(key))
                          
                          
 
      # Plotting the only after 2 epochs
        if len(self.hist['loss'])>1:
            # use clear_output(wait=True) on notebook
            N = np.arange(0, len(self.hist["loss"]))
            plt.style.use("seaborn")
            plt.figure(figsize=(12,12))

            # the number of plots are no. of metrics + the plot for losses  
            num_plots = len(self.metrics) + 1
            # initializing the subplots
            ax = [0] * num_plots
            # plotting loss
            ax[0] = plt.subplot(num_plots, 1, 1)
            ax[0].plot(N, self.hist['loss'], label = "train_loss")
            ax[0].plot(N, self.hist['val_loss'], label = "val_loss")
            ax[0].set_title(f"Training LOSS for epoch {epoch}")
            ax[0].set_xlabel("Epoch #")
            ax[0].set_ylabel("Loss")
            ax[0].legend(loc = "upper left")

            # plotting metrics
            i = 2
            for metric in self.metrics:
                ax[i-1] = plt.subplot(num_plots, 1, i)
                ax[i-1].plot(N, self.hist[metric], label = metric)
                ax[i-1].plot(N, self.hist[f"val_{metric}"], label= f"val_{metric}")
                ax[i-1].set_title(f"Training {metric.upper()} for epoch {epoch}")
                ax[i-1].set_xlabel("Epoch #")
                ax[i-1].set_ylabel(metric)
                ax[i-1].legend(loc = "upper left")
                i = i+1
            
            plt.subplots_adjust(bottom = 0.1, top=0.9, wspace=0.4, hspace=0.4)
            # Use plt.show() for displaying in notebook
            # saving the plot in output2 folder
            plt.savefig(f"./output2/{epoch}")
            plt.close()