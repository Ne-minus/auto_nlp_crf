## Ипользование  

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
### Тональность категорий
