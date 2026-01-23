import json
import requests
from requests.exceptions import ConnectionError

API_URL = "http://127.0.0.1:8000/predict"

def print_api_response(data: dict):
    print("\nAPI Response:")
    print(json.dumps(data, indent=4))
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