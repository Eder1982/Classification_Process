import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as pltcolors
from scipy.stats import kurtosis, skew
import os
import librosa, librosa.display
from scipy import stats
class Data_Base(object):
    def __init__(self):
        pass
    def Prepare_Data_to_ML(self,df,params):
        df = df.set_index(params[0])
        df=df.reset_index(drop= True)
        df = df.drop([params[1]], axis=1)
        df=df.sample(frac=1)
        return df

    def Correlations_c(self,dfx,not_use):
        print("Matriz de Correlaciones")
        if len(not_use) != 0:
            df_new= dfx.drop(not_use, axis=1)
            corr = df_new.corr()
        else:
            df_new=dfx
            corr = dfx.corr()
        return corr,df_new
    
    def Correlations(self, dfx, not_use,namex="Demo"):
        if len(not_use) != 0:
            df_new = dfx.drop(not_use, axis=1)
            numeric_columns = df_new.select_dtypes(include=['number']).columns
            corr = df_new[numeric_columns].corr()
        else:
            df_new = dfx
            numeric_columns = dfx.select_dtypes(include=['number']).columns
            corr = dfx[numeric_columns].corr()
        styled_corr =corr.style.background_gradient()
        #Save corr
        plt.figure(figsize=(10, 6))
        plt.imshow(styled_corr.data, aspect='auto', cmap='RdYlBu_r')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        plt.title("Correlations")
        c_destino="output"
        ruta_destino = os.path.join(os.getcwd(),c_destino, f"Corr_{namex}.png")
        plt.savefig(ruta_destino)
        plt.close()  # Cierra el gráfico actual antes de crear el siguiente
        return corr, df_new  
   
    def Outliers_Clean_Base(self,df,target_obj,target_list,numx):
        df_temp= df[df[target_obj]==numx].copy()
        df_temp_2= df[df[target_obj]!=numx].copy()
        df_temp_r= self.Outlier_Analysis(df_temp,target_list)
        df_over= pd.concat([df_temp_2,df_temp_r],axis=0)
        #sns.catplot(x="TYPE",data=df_over,kind='count')
        return df_over

    def Balance_SubSample(self,df,nums):
        count_clase=df[nums].value_counts()
        count_clase=count_clase.sort_values(ascending = False)
        # Dividiendo las clases
        df_over=df[df[nums]==count_clase.index[0]]
        for i in range(0,count_clase.size):
            if i+1 != count_clase.index[0]:
                df_temp= df[df[nums]==i+1]
                df_temp_under= df_temp.sample(count_clase[count_clase.index[0]],replace=True)
                df_over= pd.concat([df_temp_under,df_over],axis=0)
        #sns.catplot(x="TYPE",data=df_over,kind='count')        
        return df_over

    def Outlier_Analysis(self,data,target):
        for xena in data.columns:
            if xena in target:
                Q1 = data[xena].quantile(0.25)
                Q3 = data[xena].quantile(0.75)
                IQR = Q3 - Q1
                LR = Q1 -(1.5 * IQR)
                UR = Q3 + (1.5 * IQR)
                data.drop(data[(data[xena] > UR) | (data[xena]< LR)].index, inplace=True)
        return data

    def Analysis_Indep(self,obj,namex):
        df_temp = obj[obj['CLASE']== namex].copy()
        df_temp=df_temp.drop(columns= ['CLASE'])
        for i_elm in df_temp.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_temp[i_elm],stat="density", common_norm=False)
            plt.title(f"{namex} - {i_elm}") 
            c_destino="output"
            ruta_destino = os.path.join(os.getcwd(),c_destino, f"_{namex}_{i_elm}.png")
            plt.savefig(ruta_destino)
            plt.close()
   
    def Analysis_byAtrr(self, obj, atr, lst_p):
        plt.figure(figsize=(10, 6))
        for i in lst_p:
            f_temp = obj[obj['CLASE'] == i].copy()
            sns.histplot(data=f_temp[atr])
        c_destino="output"
        ruta_destino = os.path.join(os.getcwd(),c_destino, f"{atr}_{i}.png")
        plt.title(atr)
        plt.legend(labels=lst_p)
        plt.savefig(ruta_destino)
        plt.close()  # Cierra el gráfico actual antes de crear el siguiente

    def Multi_Picture(self,df,lst_p,lst_a):
        for i in lst_p:
            print("Analisis de {}".format(i))
            self.Analysis_Indep(df,i)
        for j in lst_a:
            print("Analisis de {}".format(j))
            self.Analysis_byAtrr(df, j, lst_p)

    def Read_csv_dataset(self,rut_file):
        # rut_file = 'D:\\0.0 Software\\2.0 Programming\\6.0 Phyton\\Data_Science\\dataset2.csv'
        df = pd.read_csv(rut_file)
        return df
    def Add_type_to_dataset(self,df):
        df['TYPE'] = df['CLASE'].map({'eslogan1': 1,'eslogan2': 2,'eslogan3': 3,'eslogan4': 4,'eslogan5': 5,'eslogan6': 6,
                                      'eslogan7': 7,'eslogan8': 8,'eslogan9': 9, 'eslogan10': 10,'otros': 11})
        return df
    def Generate_dataset(self,path):
        dataset = pd.DataFrame(columns=['ID','Media','Media absoluta','Mediana','Mediana absoluta','STD','Var','Kurtosis','Skewness',
                  'Cruces por cero','Percentil 0.0','Percentil 0.1','Percentil 0.2','Percentil 0.3','Percentil 0.4','Percentil 0.5',
                  'Percentil 0.6','Percentil 0.7','Percentil 0.8','Percentil 0.9','Percentil 1.0','Media Fourier', 'Media absoluta Fourier',
                  'Mediana Fourier','Mediana absoluta Fourier','STD Fourier','Var Fourier','Kurtosis Fourier','Skewness Fourier','CLASE'])
        dataset = dataset.set_index('ID')
        #path = "audios/audios_etiquetados/"
        files = os.listdir(path)
        for file in files:
            data, fs = librosa.load(path + file)
            #CARACTERISTICAS ESTADISTICAS DE LA SEÑAL EN EL DOMINIO TIEMPO
            #media
            mean = np.mean(data)
            #media absoluta
            meanAbs = np.mean(np.abs(data - np.mean(data)))
            #mediana
            meadian = np.median(data)
            #mediana absoluta
            meadianAbs = stats.median_abs_deviation(data, scale = 1)
            #Desviacion estandar
            std = np.std(data)
            #varianza
            var = std / mean
            #kurtosis
            kurtosisX = kurtosis(data)
            #skewness
            skewnessX = skew(data)
            #cruce por cero
            zcrs = sum(librosa.core.zero_crossings(data))
            #percentil(0.0)
            per00 = np.percentile(data,0)
            #percentil(0.1)
            per01 = np.percentile(data,10)
            #percentil(0.2)
            per02 = np.percentile(data,20)
            #percentil(0.3)
            per03 = np.percentile(data,30)
            #percentil(0.4)
            per04 = np.percentile(data,40)
            #percentil(0.5)
            per05 = np.percentile(data,50)
            #percentil(0.6)
            per06 = np.percentile(data,60)
            #percentil(0.7)
            per07 = np.percentile(data,70)
            #percentil(0.8)
            per08 = np.percentile(data,80)
            #percentil(0.9)
            per09 = np.percentile(data,90)
            #percentil(1.0)
            per10 = np.percentile(data,100)
            #CARACTERISTICAS EN EL DOMINIO DE LA FRECUENCIA
            #OBTENEMOS LA TRANSFORMADA DISCRETA DE FOURIER (DFT) MEDIANTE EL ALGORITMO FAST FOURIER TRANSFORM (FFT)
            fourier = np.fft.fft(data)
            fourier_abs = np.abs(fourier) #modulo de cada componente complejo
            #media
            mean_f = np.mean(fourier_abs)
            #media absoluta
            meanAbs_f = np.mean(np.abs(fourier_abs - np.mean(fourier_abs)))
            #mediana
            meadian_f = np.median(fourier_abs)
            #mediana absoluta
            meadianAbs_f = stats.median_abs_deviation(fourier_abs, scale = 1)
            #Desviacion estandar
            std_f = np.std(fourier_abs)
            #varianza
            var_f = std_f / mean_f
            #kurtosis
            kurtosisX_f = kurtosis(fourier_abs)
            #skewness
            skewnessX_f = skew(fourier_abs)
            #obtener la etiqueta
            index_ultimo_ = file.rfind('_')
            clase = file[index_ultimo_ + 1 : file.rfind('.') ] # a:b, no incluye el b
            #AGREGANDO AL DATASET
            row = pd.Series({
                'Media' : mean,
                'Media absoluta' : meanAbs,
                'Mediana' : meadian,
                'Mediana absoluta' : meadianAbs,
                'STD' : std,
                'Var' : var,
                'Kurtosis' : kurtosisX,
                'Skewness' : skewnessX,
                'Cruces por cero' : zcrs,
                'Percentil 0.0' : per00,
                'Percentil 0.1' : per01,
                'Percentil 0.2' : per02,
                'Percentil 0.3' : per03,
                'Percentil 0.4' : per04,
                'Percentil 0.5' : per05,
                'Percentil 0.6' : per06,
                'Percentil 0.7' : per07,
                'Percentil 0.8' : per08,
                'Percentil 0.9' : per09,
                'Percentil 1.0' : per10,
                'Media Fourier' : mean_f,
                'Media absoluta Fourier' : meanAbs_f,
                'Mediana Fourier' : meadian_f,
                'Mediana absoluta Fourier' : meadianAbs_f,
                'STD Fourier' : std_f,
                'Var Fourier' : var_f,
                'Kurtosis Fourier' : kurtosisX_f,
                'Skewness Fourier' : skewnessX_f,
                'CLASE' : clase
            }, name = file)
            dataset = dataset.append(row)
        return dataset