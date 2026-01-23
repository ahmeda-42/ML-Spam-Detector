import requests

API_URL = "http://127.0.0.1:8000/predict"

def print_api_response(data: dict):
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

def main():
    print("Interactive API Tester")
    print("Type a message and press Enter")
    print("Type 'exit' or Ctrl+C to quit")

    while True:
        try:
            message = input("\nMessage> ").strip()

            if message.lower() in {"exit", "quit"}:
                print("\n👋 Exiting.")
                break
            if not message:
                continue

            response = requests.post(
                API_URL,
                json={"message": message},
                timeout=5
            )
            response.raise_for_status()

            print_api_response(response.json())

        except ConnectionError:
            print("\nCould not connect to API.")
            print("Start it with: uvicorn app.main:app --reload\n")
            break

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

        except Exception as e:
            print(f"\n⚠️ Error: {e}\n")

if __name__ == "__main__":
    main()