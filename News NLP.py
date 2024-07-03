import requests
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import datetime
import os
import openai

openai.api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXX'
api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXX'

decision = 'train'  # "train" or "process"

model_save_path = 'C:/Users/jack/Coding Shit/Stock Project/Models/stock_performance_nlp_model' + str(datetime.datetime.today().date())
previous_model_path = 'C:/Users/jack/Coding Shit/Stock Project/Models/stock_performance_nlp_model' + str(datetime.datetime.today().date())  # path to a previously saved model

def fetch_news(api_key, query='stock'):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    articles = data['articles']
    return articles

def preprocess_articles(articles):
    texts = [article['content'] for article in articles if article['content']]
    return texts

def tokenize_function(examples, tokenizer, max_length=512):
    model_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

def summarize_articles_gpt2(texts):
    summarizer = pipeline('summarization', model='gpt2', tokenizer='gpt2')
    summaries = [summarizer(text, max_length=150, min_length=30, do_sample=False) for text in texts]
    summarized_texts = [summary[0]['summary_text'] for summary in summaries]
    return summarized_texts

def custom_score_with_llm(summary):
        prompt = (
            f"Evaluate the following summary for its level of positivity on a scale of 1 to 10, "
            f"where 1 is very negative and 10 is very positive:\n\n{summary}\n\nScore:"
        )
        
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=30,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        try:
            score = int(response.choices[0].text.strip())
            return score
        except (ValueError, IndexError):
            # Default to a neutral score if parsing fails
            return 5
        
def advanced_summarize_and_score(texts, model, tokenizer):
    summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
    summaries = [summarizer(text, max_length=150, min_length=30, do_sample=False) for text in texts]
    summarized_texts = [summary[0]['summary_text'] for summary in summaries]
    
    scores = [custom_score_with_llm(summary) for summary in summarized_texts]
    return summarized_texts, scores


if decision == 'train':
    # Gather articles
    articles = fetch_news(api_key)
    texts = preprocess_articles(articles)

    df = pd.DataFrame(texts, columns=['text'])

    # Split data into training and evaluation sets
    train_size = int(0.8 * len(df))
    train_dataset = df[:train_size]
    eval_dataset = df[train_size:]

    # Convert to Hugging Face dataset format
    train_dataset = Dataset.from_pandas(train_dataset)
    eval_dataset = Dataset.from_pandas(eval_dataset)

    # Load tokenizer and model
    model_name = 'facebook/bart-large-cnn'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Option to load a previously saved model or start from scratch
    if os.path.exists(previous_model_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(previous_model_path)
        print("Loaded model from previous checkpoint.")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Loaded pre-trained model.")

    # Tokenizing the datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Set format for PyTorch
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
        evaluation_strategy='epoch',
        save_total_limit=2,  # Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir.
    )

    rouge = load_metric('rouge', trust_remote_code=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Ensure predictions are in the correct format for batch_decode
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return result
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Ensure predictions are in the correct format for batch_decode
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=-1)  # Ensure predictions are token IDs
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

    # Initial Sentiment Analysis with GPT-2
    sentiment_analysis = pipeline('sentiment-analysis')
    initial_sentiments = sentiment_analysis(texts)
    positive_texts = [text for text, sentiment in zip(texts, initial_sentiments) if sentiment['label'] == 'POSITIVE']
    positive_articles = [article for article, sentiment in zip(articles, initial_sentiments) if sentiment['label'] == 'POSITIVE']

    print(f'Positive Texts: {positive_texts[:10]}')
    print(f'Positive Articles: {positive_articles[:10]}')

    # Advanced Summarization and Scoring
    advanced_model_name = 'facebook/bart-large-cnn'
    advanced_tokenizer = AutoTokenizer.from_pretrained(advanced_model_name)
    advanced_model = AutoModelForSeq2SeqLM.from_pretrained(advanced_model_name)

    summarizer = pipeline('summarization', model=advanced_model, tokenizer=advanced_tokenizer)
    summaries = [summarizer(text, max_length=150, min_length=30, do_sample=False) for text in positive_texts]
    summarized_texts = [summary[0]['summary_text'] for summary in summaries]

    scores = [custom_score_with_llm(summary) for summary in summarized_texts]

    # Extract company titles and other information from the articles
    titles = [article['title'] for article in positive_articles if article['title']]
    published_dates = [article['publishedAt'] for article in positive_articles]
    urls = [article['url'] for article in positive_articles]

    # Create DataFrame and save to CSV
    output_df = pd.DataFrame({
        'Company Title': titles,
        'Published Date': published_dates,
        'URL': urls,
        'Summary': summarized_texts,
        'Sentiment': ['POSITIVE'] * len(summarized_texts),
        'Promising Score': scores
    })

    most_promising = output_df[output_df['Promising Score'] >= 7].sort_values(by='Promising Score', ascending=False)

    output_csv = 'C:/Users/jack/Coding Shit/Stock Project/Data/Sentiment Analysis/StockSentiments-' + str(datetime.datetime.today().date()) + '.csv'
    output_df.to_csv(output_csv, index=False)

    print('Predictions saved to ' + str(output_csv))
    print('Most Promising Companies:')
    print(most_promising[['Company Title', 'Promising Score', 'Summary', 'URL']])



if decision == 'process':
    # Load the trained model and tokenizer for advanced analysis
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path)
    
    # Fetch new articles
    articles = fetch_news(api_key)
    texts = preprocess_articles(articles)
    
    # Initial Sentiment Analysis with GPT-2
    sentiment_analysis = pipeline('sentiment-analysis')
    initial_sentiments = sentiment_analysis(texts)
    positive_texts = [text for text, sentiment in zip(texts, initial_sentiments) if sentiment['label'] == 'POSITIVE']
    positive_articles = [article for article, sentiment in zip(articles, initial_sentiments) if sentiment['label'] == 'POSITIVE']

    print(f'Positive Texts: {positive_texts[:10]}')
    print(f'Positive articles: {positive_articles[:10]}')
    
    # Advanced Summarization and Scoring
    advanced_model_name = 'facebook/bart-large-cnn'
    advanced_tokenizer = AutoTokenizer.from_pretrained(advanced_model_name)
    advanced_model = AutoModelForSeq2SeqLM.from_pretrained(advanced_model_name)
    
    summarizer = pipeline('summarization', model=advanced_model, tokenizer=advanced_tokenizer)
    summaries = [summarizer(text, max_length=150, min_length=30, do_sample=False) for text in positive_texts]
    summarized_texts = [summary[0]['summary_text'] for summary in summaries]
    
    scores = [custom_score_with_llm(summary) for summary in summarized_texts]
    
    # Extract company titles and other information from the articles
    titles = [article['title'] for article in positive_articles if article['title']]
    published_dates = [article['publishedAt'] for article in positive_articles]
    urls = [article['url'] for article in positive_articles]
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame({
        'Company Title': titles,
        'Published Date': published_dates,
        'URL': urls,
        'Summary': summarized_texts,
        'Sentiment': ['POSITIVE'] * len(summarized_texts),
        'Promising Score': scores
    })
    
    most_promising = output_df[output_df['Promising Score'] >= 7].sort_values(by='Promising Score', ascending=False)
    
    output_csv = 'C:/Users/jack/Coding Shit/Stock Project/Data/Sentiment Analysis/StockSentiments-' + str(datetime.datetime.today().date()) + '.csv'
    output_df.to_csv(output_csv, index=False)
    
    print('Predictions saved to ' + str(output_csv))
    print('Most Promising Companies:')
    print(most_promising[['Company Title', 'Promising Score', 'Summary', 'URL']])

