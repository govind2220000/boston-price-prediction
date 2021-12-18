import pickle

class model_save:
    
    def save_model(self, model_object,model_name):
        pickle.dump(model_object, open(f'src/Models/{model_name}.pkl', 'wb'))
# with open(f'src/Models/{best_model}.pkl', 'wb') as f:
#     pickle.dump(model_dict[i], f)