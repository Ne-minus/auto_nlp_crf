from collections import defaultdict


class Evaluator:
    def __init__(self, path_aspects, path_categories):
        self.gold_aspects = path_aspects
        self.gold_cats = path_categories

    
    def reprocess_gold(self):
        gold_aspect_cats = {}
        with open(self.gold_aspects) as fg:
            for line in fg:
                line = line.rstrip('\r\n').split('\t')
                if line[0] not in gold_aspect_cats:
                    gold_aspect_cats[line[0]] = {"starts":[], "ends":[], "cats":[], "sents":[]}
                gold_aspect_cats[line[0]]["starts"].append(int(line[3]))
                gold_aspect_cats[line[0]]["ends"].append(int(line[4]))
                gold_aspect_cats[line[0]]["cats"].append(line[1])
                gold_aspect_cats[line[0]]["sents"].append(line[5])
        
        return gold_aspect_cats

    def aspects_scores(self, pred_aspects_path):
        gold_aspect_cats = self.reprocess_gold()
        full_match, partial_match, full_cat_match, partial_cat_match = 0, 0, 0, 0
        total = 0
        fully_matched_pairs = []
        partially_matched_pairs = []
        with open(pred_aspects_path) as fp:
            for line in fp:    
                total += 1
                line = line.rstrip('\r\n').split('\t')
                start, end = int(line[3]), int(line[4])
                category = line[1]
                doc_gold_aspect_cats = gold_aspect_cats[line[0]]
                if start in doc_gold_aspect_cats["starts"]:
                    i = doc_gold_aspect_cats["starts"].index(start)
                    if doc_gold_aspect_cats["ends"][i] == end:
                        full_match += 1
                        if doc_gold_aspect_cats["cats"][i] == category:
                            full_cat_match += 1
                        else:
                            partial_cat_match += 1
                        fully_matched_pairs.append(
                            (
                                [
                                    doc_gold_aspect_cats["starts"][i], 
                                    doc_gold_aspect_cats["ends"][i], 
                                    doc_gold_aspect_cats["cats"][i],
                                    doc_gold_aspect_cats["sents"][i]
                                ],
                                line
                            )
                        )
                        continue
                for s_pos in doc_gold_aspect_cats["starts"]:
                    if start <= s_pos:
                        i = doc_gold_aspect_cats["starts"].index(s_pos)
                        if doc_gold_aspect_cats["ends"][i] == end:
                            partial_match += 1
                            partially_matched_pairs.append(
                                (
                                    [
                                        doc_gold_aspect_cats["starts"][i], 
                                        doc_gold_aspect_cats["ends"][i], 
                                        doc_gold_aspect_cats["cats"][i],
                                        doc_gold_aspect_cats["sents"][i]
                                    ],
                                    line
                                )
                            )
                            if doc_gold_aspect_cats["cats"][i] == category:
                                partial_cat_match += 1
                            continue
                        matched = False
                        for e_pos in doc_gold_aspect_cats["ends"][i:]:
                            if s_pos <= end <= e_pos:
                                partial_match += 1
                                partially_matched_pairs.append(
                                    (
                                        [
                                            doc_gold_aspect_cats["starts"][i], 
                                            doc_gold_aspect_cats["ends"][i], 
                                            doc_gold_aspect_cats["cats"][i],
                                            doc_gold_aspect_cats["sents"][i]
                                        ],
                                        line
                                    )
                                )
                                if doc_gold_aspect_cats["cats"][i] == category:
                                    partial_cat_match += 1
                                matched = True
                                break
                        if matched:
                            break
                    if start > s_pos:
                        i = doc_gold_aspect_cats["starts"].index(s_pos)
                        if start < doc_gold_aspect_cats["ends"][i] <= end:
                            partial_match += 1
                            partially_matched_pairs.append(
                                (
                                    [
                                        doc_gold_aspect_cats["starts"][i], 
                                        doc_gold_aspect_cats["ends"][i], 
                                        doc_gold_aspect_cats["cats"][i],
                                        doc_gold_aspect_cats["sents"][i]
                                    ],
                                    line
                                )
                            )
                            if doc_gold_aspect_cats["cats"][i] == category:
                                partial_cat_match += 1
                            break

        gold_size = sum([len(gold_aspect_cats[x]["cats"]) for x in gold_aspect_cats])


        full_match_pr = full_match / total
        full_mathch_re = full_match / gold_size
        partial_match_ration = (full_match + partial_match)  / total
        full_cat_acc = full_cat_match / total
        partial_cat_acc = (full_cat_match + partial_cat_match) / total


        return (full_match_pr, full_mathch_re, partial_match_ration, full_cat_acc, partial_cat_acc), fully_matched_pairs, partially_matched_pairs
    
    def sentiment_accuracy(self, matches):
        matched_sentiment = 0.
        for pair in matches:
            *_, gold_s = pair[0]
            *_, pred_s = pair[1]
            if gold_s == pred_s:
                matched_sentiment += 1

        return matched_sentiment / len(matches)
    
    def sentiment_cats(self, pred_cats):
        with open(self.gold_cats) as gc, open(pred_cats) as pc:
            gold_labels = set(gc.readlines())
            pred_labels = set(pc.readlines())

            return len(gold_labels & pred_labels) / len(gold_labels)


if __name__ == '__main__':
    gold_aspects = input('Path to gold aspects: ')
    if not gold_aspects:
        gold_aspects = './data/aspects_dev.csv'

    pred_aspects = input('Path to predicted aspects: ')
    if not pred_aspects:
        pred_aspects = './data/res/aspects_pred.csv'

    gold_cats = input('Path to gold categories: ')
    if not gold_cats:
        gold_cats = './data/categories_dev.csv'

    pred_cats = input('Path to predicted categories: ')
    if not pred_cats:
        pred_cats = './data/res/categories_dev.csv'

    evaluation = Evaluator(gold_aspects, gold_cats)

    print('Aspect extraction results: ')
    metrics, fully_matched, partially_matched = evaluation.aspects_scores(pred_aspects)
    print(f"""
        Full match precision: {metrics[0]}
        Full match recall: {metrics[1]}
        Partial match ratio in pred: {metrics[2]}
        Full category accuracy: {metrics[3]}
        Partial category accuracy: {metrics[4]}
        """)

    print('Sentiment analysis results: ')
    sentiment_full = evaluation.sentiment_accuracy(fully_matched)
    sentiment_part = evaluation.sentiment_accuracy(partially_matched)
    print(f'''
        Mention sentiment accuracy on full matches:{sentiment_full}
        Mention sentiment accuracy on partial matches:{sentiment_part}
        ''')
    
    # print(f'Overall accuracy by categories: {evaluation.sentiment_cats(pred_cats)}')

          


