# coding=utf-8

from collections import Counter

def get_entities_with_bios(seq):
    """
    Get entities from sequence tagged with BIOS.
    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        >>> get_entities_with_bios(seq)
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for index, tag in enumerate(seq):
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[2] = index
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        elif tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities_with_bio(seq):
    """
    Get entities from sequence tagged with BIO.
    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entity_bio(seq)
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for index, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[0] = tag.split('-')[1]
            chunk[1] = index
            chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, tag='bios'):
    assert tag in ['bio', 'bios']
    if tag == 'bio':
        return get_entities_with_bio(seq)
    else:
        return get_entities_with_bios(seq)


class EntityScore:
    def __init__(self, id2label, tag='bios'):
        self.id2label = id2label
        self.tag = tag
        self.reset()
    
    def reset(self):
        self.trues = []
        self.preds = []
        self.rights = []
    
    def update(self, true_labels, pred_labels):
        """
        Example:
            >>> true_labels = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_labels = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        """
        for true_list, pred_list in zip(true_labels, pred_labels):
            if not isinstance(true_list[0], str):
                true_list = [self.id2label[_] for _ in true_list]
            if not isinstance(pred_list[0], str):
                pred_list = [self.id2label[_] for _ in pred_list]
            
            true_entities = get_entities(true_list, tag=self.tag)
            pred_entities = get_entities(pred_list, tag=self.tag)
            self.trues.extend(true_entities)
            self.preds.extend(pred_entities)
            self.rights.extend([entity for entity in pred_entities if entity in true_entities])

    def result(self):
        class_info = {}
        true_counter = Counter([chunk[0] for chunk in self.trues])
        pred_counter = Counter([chunk[0] for chunk in self.preds])
        right_counter = Counter([chunk[0] for chunk in self.rights])
        for _type, count in true_counter.items():
            true = count
            pred = pred_counter.get(_type, 0)
            right = right_counter.get(_type, 0)
            precision, recall, f1 = self.compute(true, pred, right)
            class_info[_type] = {"acc": precision, 'recall': recall, 'f1': f1}
        
        true = len(self.trues)
        pred = len(self.preds)
        right = len(self.rights)
        precision, recall, f1 = self.compute(true, pred, right)
        total_info = {"acc": precision, 'recall': recall, 'f1': f1}
        return (total_info, class_info)
    
    def compute(self, true, pred, right):
        precision = 0 if pred == 0 else right / pred
        recall = 0 if true == 0 else right / true
        f1 = 0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall, f1
