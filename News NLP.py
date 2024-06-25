import requests
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import datetime

api_key = '682be6864fe5485bbdd773141bcae908'

decision = 'train'  # "train" or "process" to decide if new model or analyzing new articles

model_save_path = 'C:/Users/jack/Coding Shit/Stock Project/Models/stock_performance_nlp_model'

def fetch_news(api_key, query='stock'):
    url = 'https://newsapi.org/v2/everything?q=' + str(query) + '&apiKey=' + str(api_key)
    response = requests.get(url)
    data = response.json()
    print(data)
    articles = data['articles']
    return articles

def preprocess_articles(articles):
    texts = [article['content'] for article in articles if article['content']]
    return texts

# load and tokenize data fuction
def tokenize_function(examples, tokenizer, max_length=512):
    model_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

if decision == 'train':
    articles = fetch_news(api_key)
    texts = preprocess_articles(articles)

    df = pd.DataFrame(texts, columns=['text'])

    # splitting
    train_size = int(0.8 * len(df))
    train_dataset = df[:train_size]
    eval_dataset = df[train_size:]

    # converting to Hugging Face dataset
    train_dataset = Dataset.from_pandas(train_dataset)
    eval_dataset = Dataset.from_pandas(eval_dataset)

    # load tokenizer/model
    model_name = 'facebook/bart-large-cnn'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # tokenizing
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # setting pytorch format
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    rouge = load_metric('rouge', trust_remote_code=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if decision == 'process':
    # loading model
    model_name = model_save_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # gather articles
    articles = fetch_news(api_key)
    texts = preprocess_articles(articles)
    
    summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
    summaries = [summarizer(text, max_length=150, min_length=30, do_sample=False) for text in texts]
    summarized_texts = [summary[0]['summary_text'] for summary in summaries]
    
    # calling analysis 
    sentiment_analysis = pipeline('sentiment-analysis')
    results = sentiment_analysis(summarized_texts)
    
    # Extract company titles (for demonstration, assuming titles are in the articles)
    titles = [article['title'] for article in articles if article['title']]
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame({
        'Company Title': titles,
        'Summary': summarized_texts,
        'Sentiment': [result['label'] for result in results],
        'Sentiment Score': [result['score'] for result in results]
    })
    
    output_csv = 'C:/Users/jack/Coding Shit/Stock Project/Data/Sentiment Analysis/StockSentiments-' + str(datetime.datetime.today().date()) + '.csv'
    output_df.to_csv(output_csv, index=False)
    print('Predictions saved to ' + str(output_csv))
