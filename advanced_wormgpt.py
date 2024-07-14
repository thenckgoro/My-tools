from transformers import pipeline

# Load pre-trained models for different tasks
text_classifier = pipeline("text-classification")
summarizer = pipeline("summarization")
text_generator = pipeline("text-generation")

def classify_text(text):
    results = text_classifier(text)
    for result in results:
        print(f"Label: {result['label']}, Confidence: {result['score']:.2f}")

def summarize_text(text):
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    print("Summary:", summary[0]['summary_text'])

def generate_text(prompt):
    generated_text = text_generator(prompt, max_length=100, num_return_sequences=1)
    print("Generated Text:", generated_text[0]['generated_text'])

if __name__ == "__main__":
    while True:
        print("\nChoose an option:")
        print("1. Classify Text")
        print("2. Summarize Text")
        print("3. Generate Text")
        print("4. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            text = input("Enter text to classify: ")
            classify_text(text)
        elif choice == '2':
            text = input("Enter text to summarize: ")
            summarize_text(text)
        elif choice == '3':
            prompt = input("Enter text to generate: ")
            generate_text(prompt)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")
