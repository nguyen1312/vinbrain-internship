from lib import *
from glob import glob

def preprocessing(df):
    train_fns = sorted(glob('./siim/dicom-images-train/*/*/*.dcm'))
    missing = 0
    multiple = 0
    patients_data = []
    for k,paths in enumerate(train_fns):
        patient = {}
        img_id = paths.split('/')[-1]
        data = pydicom.dcmread(paths)
        try:
            tmp = df[df['ImageId'] == '.'.join(img_id.split('.')[:-1])]
            
            if tmp.shape[0] > 1: 
                multiple += 1
            rle = tmp['EncodedPixels'].values
            if rle[0] == '-1':
                pixels = rle[0]
            else:    
                pixels = [i for i in rle]
            patient["UID"] = data.SOPInstanceUID
            patient['EncodedPixels'] = pixels
            patient["filepath"] = paths
            patients_data.append(patient)
        except:
            missing += 1

    df_patients = pd.DataFrame(patients_data, columns=["UID", "EncodedPixels", "Age", 
                                "Sex", "Modality", "BodyPart", "ViewPosition", "filepath"])
    df_patients['Pneumothorax'] = df_patients['EncodedPixels'].apply(lambda x:0 if x == '-1' else 1)
    df_patients['Pneumothorax'] = df_patients['Pneumothorax'].astype('int')
    return df_patients

if __name__ == "__main__":
    df = pd.read_csv('./siim/train-rle.csv')
    df_new = preprocessing(df)
    df_new.to_csv('preprocessing_data.csv', index = False)
    print(df_new.head())
   
    