from data_base import Data_Base
from NeuronalNetwork import RedNeuronal

def main():
    #CREANDO NUEVO CÓDIGO POR EDER
    rut1 = 'dataset.csv'
    db=Data_Base()
    df=db.Read_csv_dataset(rut1) # Convierte el file en dataframe
    df= db.Add_type_to_dataset(df) # Agrega type a dataframe
    # Data Analysis of Correlations
    corr,df= db.Correlations(df,[],"Ini")
    # Selected good attributes. 
    # The analsys of Pearson Correlations show us that de a8 parameters is not good for to predictibility
    # Data Analysis of Correlations with data that is not necesary
    target= "TYPE"
    not_use = ['Percentil 0.1','Kurtosis Fourier','Percentil 0.0','Mediana Fourier','Mediana absoluta Fourier','Media absoluta Fourier','STD Fourier','Media absoluta','STD','Percentil 0.2', 'Percentil 0.3', 'Percentil 0.4','Percentil 0.5','Percentil 0.6', 'Percentil 0.7', 'Percentil 0.8','Percentil 0.9']
    corr,df= db.Correlations(df,not_use,"End")
    # Clear Ouliers
    target_obj= 'TYPE'
    target_list= ['Media', 'Mediana', 'Mediana absoluta', 'Var', 'Kurtosis',
        'Skewness', 'Cruces por cero', 'Percentil 1.0', 'Media Fourier',
        'Var Fourier', 'Skewness Fourier']
    df_balance=db.Outliers_Clean_Base(df,target_obj,target_list,11)
    # Balance of data con Sub Sumple Method
    df_balance =db.Balance_SubSample(df_balance,"TYPE")
    # Estadisticas y correlaciones
    lst_p = ["eslogan1", "eslogan2","eslogan3","eslogan4","eslogan5","eslogan6","eslogan7","eslogan8","eslogan9","eslogan10","otros"]
    lst_a = ['Media', 'Mediana', 'Mediana absoluta', 'Var', 'Kurtosis','Skewness', 'Cruces por cero', 'Percentil 1.0', 'Media Fourier','Var Fourier', 'Skewness Fourier']
    df2 = df_balance.drop(["ID","TYPE"], axis=1)
    #db.Multi_Picture(df2,lst_p,lst_a)
    #Prepare Data to ML
    df_ML=db.Prepare_Data_to_ML(df_balance,['ID',"CLASE"])
    # Create Model of Training and Testing with Red Neuronal
    rn = RedNeuronal(df_ML,target,42)
    #This method generate the best hyperparameters of Neuronal Network like activation functiona and neurones of each level
    #params=rn.GridSearch_Hyperparamters(12, 12)
    #p1,p2=rn.Indetify_Hyperparamters_Analysis(2,2)
    #print(p1) # logistic
    #print(p2) #  (1, 5)
    # Generate the best model
    activation= 'logistic'
    neu1=10
    neu2= 10
    N_SP=20
    #mod,esc= rn.Generate_Neuronal_Model(activation,neu1,neu2) # Generate model fitting and escalador
    mod= rn.Generate_Neuronal_Model_KNFold(activation,neu1,neu2,N_SP)
    # Testing with new data
    # rut_new= file del nuevo audio
    #y_pred= rn.Generate_Prediction(mod,esc,rut_new)
if __name__ == "__main__":
    main()




# Llama a la función principal si este archivo es el punto de entrada
if __name__ == "__main__":
    main()
