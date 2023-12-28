class SentimentBert:

  def __init__(self, model_name, prompt_type, lang):
    self.model = model_name

    if lang == 'ru':
      self.rubert_map = {'Whole':'впечатление вцелом','Service':'сервис',
                         'Food':'еда', 'Interior': 'интерьер',
                         'Price': 'цена'}
      self.acc_map = {'NEGATIVE': 'negative', 
                      'NEUTRAL': 'neutral', 
                      'POSITIVE': 'positive'}
    elif lang == 'en':
      self.acc_map = {'1 star': 'negative', 
                      '2 stars': 'negative', 
                      '3 stars': 'neutral', 
                      '4 stars': 'positive', 
                      '5 stars': 'positive'}
    self.prompt_type = prompt_type
  
  def prompt_types(self, prompt_type, sent, text=None, aspect=None, category=None):
    if prompt_type == 1:
      prompt = f'{sent} [SEP] {aspect}, {category}'

    elif prompt_type == 2:
      prompt = f"Определи тональность аспекта '{aspect}' в следующем тексте: '{sent}'"

    elif prompt_type == 3:
      prompt = sent

    # для тональности всей категории в отзыве
    elif prompt_type == 4:
      prompt = f'{text} [SEP] {category}'

    return prompt

  def predict(self, file):
    df = pd.read_csv(file, sep='\t')
    sentiment_analysis = pipeline('text-classification', model=self.model)
    data = []

    for _, row in df.iterrows():
      if self.prompt_type == 1:
        result = sentiment_analysis(prompt_types(self.prompt_type, row.sentence, row.aspect, row.category))
      elif self.prompt_type == 2:
        result = sentiment_analysis(prompt_types(self.prompt_type, row.sentence, row.aspect))
      elif self.prompt_type == 3:
        result = sentiment_analysis(prompt_types(self.prompt_type, row.sentence))
      elif self.prompt_type == 4:
        result = sentiment_analysis(prompt_types(self.prompt_type, row.text, row.category))

      for sentiment in result:
        label = sentiment['label']
        score = sentiment['score']

      data.append([label])
    labels = pd.DataFrame(data, columns=['sentiment'])

    pred_file = pd.concat([df, labels.sentiment.map(self.acc_map)], axis=1).to_csv('aspect_sentiment.csv')
    
    return pred_file
