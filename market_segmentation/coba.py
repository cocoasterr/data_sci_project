import pickle
import os


cur_dir = os.getcwd()
filename = f"{cur_dir}/final_model.sav"
loaded_model = pickle.load(open(filename, "rb"))
saddsa = "sad"
