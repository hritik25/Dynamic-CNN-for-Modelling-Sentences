import csv

# Will return the contents of a file in the form 
# of a list be it sentences, phrases or labels
def loadData(filename):    
    with open(filename, 'rb') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.append(row)
    return data