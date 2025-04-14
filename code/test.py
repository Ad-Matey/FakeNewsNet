import torch
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

modelToUse = input('Model to use (politifact - p, gossipcop - g, both - b): ')

# Used to convert from model prediction to actual label of data
dataConversion = {
    0 : "Real",
    1 : "Fake"
}

if (modelToUse == 'p'):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='results/politifact/politifact_model/', local_files_only=True)
elif (modelToUse == 'g'):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='results/gossipcop/gossipcop_model/', local_files_only=True)
elif (modelToUse == 'b'):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='results/combined/combined_model/', local_files_only=True)
else:
    raise ValueError("Invalid model choice")

repeat = True
while repeat:

    textInput = input('\nInput to Model: ')
    textInput = tokenizer(textInput, padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        logits = model(**textInput).logits

    predicted_class_id = logits.argmax().item()
    print(dataConversion[predicted_class_id])

    repeatInput = input('Enter another value (y/n)? ')
    if (repeatInput.lower().strip() != 'y'):
        repeat = False