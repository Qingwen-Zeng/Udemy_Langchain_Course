from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file
def main():
    print("Hello from udemy-langchain-course!")
    print(os.environ.get('OPENAI_API_KEY'))

if __name__ == "__main__":
    main()
