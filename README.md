## Ипользование  
1. Склонировать репозиторий
2. Запустить файл [**evaluation.py**](https://github.com/Ne-minus/nlp_4th_year_absa_project/blob/pipeline_interface/evaluation.py)
3. Следовать инструкциям, появляющимся в командной строке (необходимо указать пути к тестовому датасету и золотому стандарту)
```
git clone https://github.com/Ne-minus/nlp_4th_year_absa_project.git
pip install -r requirements.txt
python3 evaluation.py
```

## Задачи и методы  
### Выделение аспектов и категорий
Мы воспользовались классическим методом, изпользубщимся для решения этой задачи, -- conditional random fields (CRF).  Реализация была взята из модуля [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/). Также как и авторы туториала, мы учитывали предыдущее и последующее слово + их POS-теги.  

**Препроцессинг**
1) Токенизируем исходный текст и для каждого токена находим позицию начала и конца при помощи spacy (для обучения и для инференса)
2) Добавляем BIO разметку из тренировочного сета (только для обучения).

**Обучение**

1) Сначала мы подавали модели на вход классические теги nltk (не UD) и получили следующие результаты (веса [тут](./checkpoints/crf_weights.sav)):

| Метрика       |               | 
| :------------- |:------------------:|
| Full-match precision   | 0.71    |
|Full-match recall   | 0.46 |
| Partial match ratio  | 0.92        |
| Full category accuracy  | 0.75         |
| Partial category accuracy  | 0.92         |

Обнаруженные проблемы:
- в автоматической BIO разметке встречаются I- теги без тега начала именованной сущности. В таких случаях первый I- заменяли на B- (I-whole I-whole --> B-whole I-whole)
- нас смутили теги, поэтому мы решили попробовать universal tag set (веса [тут](./checkpoints/crf_weights_ud+positions.sav))

2) Полученный результат:

| Метрика       |               | 
| :------------- |:------------------:|
| Full-match precision   | **0.9**   |
|Full-match recall   | 0.31 |
| Partial match ratio  | **0.96**        |
| Full category accuracy  | **0.9**         |
| Partial category accuracy  | **0.94**         |


Качество заметно выросло, особенно для full category accuracy, поэтому решили остановиться на варианте с UD-тегами.

### Анализ тональности
Первым этапом анализа тональностей в нашей работе было использование готовых моделей для задачи text-classification: [model 1](https://huggingface.co/MonoHime/rubert-base-cased-sentiment-new) – для русского языка и [model 2](https://huggingface.co/marianna13/bert-multilingual-sentiment) – мультиязычная.

Мы также ставили эксперименты с типом инпута для моделей:
1. prompt = f'{sent} [SEP] {aspect}, {category}'
2. prompt = f"Определи тональность аспекта '{aspect}' в следующем тексте: '{sent}'"
3. prompt = исходному предложению, где упоминается аспект

Однако по всем видам значение accuracy было не показательным:
| Метрика       |               | 
| :------------- |:------------------:|
| Mention sentiment accuracy on full matces   | 0.67    |
|Mention sentiment accuracy on partial matces   | 0.61 |

Следующим этапом было fine-tuning собственной модели с учетом наших данных. Для этого мы использовали модель **sberbank-ai/ruBert-base**. После экспериментов с готовыми моделями мы решили взять первый вид инпута, поскольку в обеих моделях он показал сравнительно лучший результат.
Поэтому далее для обучения моедли данные приводились к данному виду **prompt = f'{sent} [SEP] {aspect}, {category}'**
Причем английские категории были переведены в аналогичные русские.

Результаты этого этапа улучшили показатель accuracy:
| Метрика       |               | 
| :------------- |:------------------:|
| Mention sentiment accuracy on full matces   | 0.81    |
|Mention sentiment accuracy on partial matces   | 0.67 |

Данные по обучению, сохранению весов и загрузки f-t модели на hf и последующее ее использование для inference: [тетрадка](./Bert_Sentiment.ipynb)

### Тональность категорий
Подсчет тональности категорий мы проиводили по приниципу -- наиболее типичный sentiment для аспектов категории формурует ее тональность. Отутствующие категории маркировались как absence.
