import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import numpy as np
np.set_printoptions (threshold=None, edgeitems=None, linewidth=None, suppress=None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def standardize(col):
   return (col - col.mean()) / col.std()

def tabnet_incident_detect_train(dataset_path):

   """
   TabNet training for incident detection
   """

   data_types = {
         'incident_edge': 'object',  # Replace 'Column_Name1' with the actual column name
         'incident_lane': 'object'  # Replace 'Column_Name2' with the actual column name
      }

   df = pd.read_csv(dataset_path,dtype=data_types)

   # IF detecting incidents
   df = df[600:]
   df =df.fillna(-1)

   Y = df["accident_label"]      
   X = df.drop(["Unnamed: 0","step","incident_edge","incident_start_time","incident_type","accident_id","accident_duration","incident_lane","accident_label"],axis=1)
   X = X.apply(standardize)

   X= X.values
   Y= Y.values
   clf = TabNetClassifier(device_name='cuda')

   print("Splitting into test train")
   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

   clf = TabNetClassifier(n_d=64, n_a=64, momentum=0.3, n_steps=5, optimizer_fn=torch.optim.Adam,
   optimizer_params=dict(lr=2e-2),
   scheduler_params = {"gamma": 0.95,
                  "step_size": 20},
   scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15)

   print("Starting Training...")
   clf.fit(
   X_train, y_train,num_workers = 10,max_epochs=80
   )

   saving_path_name = "tabnet_incident_detect"
   print("Saving model...")
   clf.save_model(saving_path_name)

   X= X_test
   Y= y_test
   y_pred = clf.predict(X)
   y_pred = list(y_pred)
   y_test = list(Y)

   accuracy = accuracy_score(y_test, y_pred)*100
   precision = precision_score(y_test, y_pred,average = "macro")
   f1 = f1_score(y_test, y_pred,average = "macro")
   recall = recall_score(y_test, y_pred,average = "macro")
   print("Accuracy: %.2f%%" % (accuracy))
   print(f"FAR {100-accuracy}")
   print("precision: %.2f%%" % (precision * 100.0))
   print("f1: %.2f%%" % (f1 * 100.0))
   print("recall: %.2f%%" % (recall * 100.0))

   return accuracy,precision,f1,recall


