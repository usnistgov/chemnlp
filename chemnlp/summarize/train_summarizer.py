#https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb#scrollTo=ZtNs9ytpCow2
# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import wandb
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
from datasets import load_metric
from tqdm import tqdm
from transformers import pipeline
from datasets import load_metric
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

metric = load_metric("rouge")
def calc_rouge_scores(candidates, references):
    result = metric.compute(predictions=candidates, references=references, use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100, 1) for key, value in result.items()}
    return result

device = 'cuda' if cuda.is_available() else 'cpu'
# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text #title
        self.ctext = self.data.ctext #abstract

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())
        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }



# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network 

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]
        
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        #if _%10 == 0:
        #    print({"Training Loss": loss.item()})

        #if _%500==0:
        #    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()



def validate(epoch, tokenizer, model, device, loader,val_max_length=25):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=val_max_length, 
                #max_length=50, 
                #max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals
import pprint
from collections import defaultdict
def main():
    # WandB – Initialize a new run
    #wandb.init(project="transformers_tutorials_summarization")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    #= wandb.config          # Initialize config
    config=defaultdict()  

    config['train_batch_size']=32
    config['val_batch_size']=16
    config['train_epochs']=10
    config['val_epochs']=1
    config['learning_rate']=1e-4
    config['seed']=42
    config['max_abstract_len']=150 #150 #150
    config['max_summary_len']=100 #100
    config['max_val_summary_len']=25 #25
    print ('Config:')
    pprint.pprint(config)
    TRAIN_BATCH_SIZE =  config['train_batch_size'] #16    # input batch size for training (default: 64)
    #TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    VALID_BATCH_SIZE = config['val_batch_size']    # input batch size for testing (default: 1000)
    #VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
    TRAIN_EPOCHS = config['train_epochs'] #50        # number of epochs to train (default: 10)
    #TRAIN_EPOCHS = 2        # number of epochs to train (default: 10)
    VAL_EPOCHS = config['val_epochs']
    LEARNING_RATE = config['learning_rate']    # learning rate (default: 0.01)
    SEED = config['seed']               # random seed (default: 42)
    #MAX_LEN: maximum length of abstracts
    #SUMMARY_LEN: maximum summarization length
    MAX_LEN = config['max_abstract_len'] #512
    #MAX_LEN = 25
    #SUMMARY_LEN = 10
    SUMMARY_LEN = config['max_summary_len']
    #SUMMARY_LEN = 150 
    val_max_length = config['max_val_summary_len']

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    #df = pd.read_csv('news_summary.csv',encoding='latin-1')
    df = pd.read_csv('../cond_mat_small.csv')
    df = pd.read_csv('../cond_mat.csv')
    #df = pd.read_csv('cond_mat_small.csv')
    #df = df[['title','abstract']]
    df['text']=df['title']
    df['ctext']=df['abstract']
    #df = df[['text','ctext']]
    df.ctext = 'summarize: ' + df.ctext
    print(df.head())

    
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state = SEED)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    train_dataset.to_csv('train_dataset.csv')
    val_dataset.to_csv('val_dataset.csv')
    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    # Log metrics with wandb
    #wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    torch.save(model.state_dict(),'model.pt')
    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader,val_max_length=val_max_length)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv('predictions.csv')
        print('Output Files generated for review')
    df=pd.read_csv('predictions.csv')
    print ('predictions')
    print(df)
    print ('Generated',df['Generated Text'][0])
    print ()
    print('Actual',df['Actual Text'][0])
    print ('Rougue T5',calc_rouge_scores(df['Actual Text'],df['Generated Text']))
    print ('Untrained baseline')

    metric = load_metric("rouge")
    # Load the summarization pipeline
    summarizer = pipeline("summarization")


    # Define the text to be summarized

    def summarz(text="", max_length=25, min_length=10):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    df=pd.read_csv('val_dataset.csv')
    df['title_split']=df['title'].apply(lambda x: len(x.split()))
    print (df.describe())
    df['summ']=df['abstract'].progress_apply(lambda x:summarz(x,max_length=val_max_length) )
    print ('Rougue baseline',calc_rouge_scores(df['title'],df['summ']))
if __name__ == '__main__':
    main()
