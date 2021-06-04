from lib import *
from dataset import SIIMDataset

# sampled dataset
def sampledDataset(df, fold):
    df_positiveCase = df[df["Pneumothorax"] == 1]
    df_negativeCase = df[df["Pneumothorax"] == 0]
    df_negativeCase_sampled = df_negativeCase.sample(len(df_positiveCase) + 1000, random_state = 2019)
    newSub_Dataframe = pd.concat([df_positiveCase, df_negativeCase_sampled])
    kfold = StratifiedKFold(n_splits = 5, random_state = 43, shuffle = True)
    df_split = newSub_Dataframe
    # k_fold validation
    train_idx, val_idx = list(kfold.split(X = df_split["UID"], y = df_split["Pneumothorax"]))[fold]
    train_df, val_df = df_split.iloc[train_idx], df_split.iloc[val_idx]
    f_names_datatrain = train_df.iloc[:, 0].values.tolist()
    f_names_dataval = val_df.iloc[:, 0].values.tolist()
    size = 512
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset = SIIMDataset(train_df, f_names_datatrain, size, mean, std, phase = "train")
    val_dataset = SIIMDataset(val_df, f_names_dataval, size, mean, std, phase = "val")
    return {
          "train": DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 4),
          "val":  DataLoader(val_dataset, batch_size = 8, shuffle = True, num_workers = 4)
    }

# Test
if __name__ == "__main__":
    df = pd.read_csv('preprocessing_data.csv')
    dataloader = sampledDataset(df, 2)
    train_dataloader = dataloader["train"]
    print(next(iter(train_dataloader)))