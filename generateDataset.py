import pytreebank
import vocabulary
import csv
import re

dataset = pytreebank.load_sst('trees/')

# I am training the DCNN for binary classification only, 
# and not fine-grained classification. In the Stanford 
# Treebank Dataset, ratings left and right of 2 denote 
# positive and negative reviews respectively, so I will 
# pick only polar reviews and add the corresponding labels. 
polarTrainingReivews = []
for example in dataset['train'][:]:    
    for newSentence in example.to_labeled_lines():
        label, sentence = newSentence
        if label != 2:
            polarTrainingReivews.append(newSentence)

polarValidationReivews = []
for example in dataset['dev'][:]:    
    newSentence = example.to_labeled_lines()[0]
    label, sentence = newSentence
    if label != 2:
        polarValidationReivews.append(newSentence)

polarTestReivews = []
for example in dataset['test'][:]:    
    newSentence = example.to_labeled_lines()[0]
    label, sentence = newSentence
    if label != 2:
        polarTestReivews.append(newSentence)

vocab = vocabulary.generateVocab('stanfordSentimentTreebank/datasetSentences.txt')

# The network-friendly dataset will be generated in the directory 'myDataset'
directory = 'myDataset/'
filenames = [('train.txt', 'train_label.txt'), ('dev.txt', 'dev_label.txt'), ('test.txt', 'test_label.txt')]
polarReviews = [polarTrainingReivews, polarValidationReivews, polarTestReivews]

for files, data in zip(filenames, polarReviews):
    labels = []
    with open(directory + files[0], 'wb') as txtfile:
        writer = csv.writer(txtfile)
        for number in xrange(len(data)):
            label = data[number][0]/3
            labels.append([label])
            breakup = [i for i in re.split(r'\s|\W', data[number][1].lower()) if i]
            # mapping the words to embeddings indices
            mappedSentence = []
            for token in breakup:
                if token in vocab:
                    mappedSentence.append(str(vocab[token]))
                else:
                    mappedSentence.append(15448)
            writer.writerow(mappedSentence)

    with open(directory + files[1], 'wb') as txtfile:
        writer = csv.writer(txtfile)
        writer.writerows(labels)

    print files, 'done!'

