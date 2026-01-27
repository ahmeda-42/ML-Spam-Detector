from model.predict import predict, predict_and_explain

MESSAGES = [
    "Win free money now",
    "Hey are you coming later?",
    "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
]

for msg in MESSAGES:
    data = predict_and_explain(msg)

    print(f'\n"{data["message"]}"')

    prediction = "SPAM" if data["prediction"]=="spam" else "NOT SPAM"
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {data['confidence']}%")

    print("\nWhy? (Percentages show relative contributions of words to the model's decision, not absolute probability)")
    for item in data["explanation"]:
        word = item["word"]
        direction = "increased spam likelihood" if item["direction"]=="spam" else "decreased spam likelihood"
        percent = item["percent"]
        print(f'• "{word}" {direction} by {percent}%')
    
    print("\n" + "="*50)