import streamlit as st
import speech_recognition as sr
import nltk
import os
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources NLTK une fois
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Charger le fichier texte et prétraiter les données
with open('DataScience.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokeniser le texte en phrases
sentences = sent_tokenize(data)

# Définir une fonction pour prétraiter chaque phrase
def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Prétraiter chaque phrase dans le texte
corpus = [preprocess(sentence) for sentence in sentences]

# Définir une fonction pour trouver la phrase la plus pertinente en fonction d'une requête
def get_most_relevant_sentence(query):
    query = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

# Fonction du chatbot
def chatbot(query):
    try:
        most_relevant_sentence = get_most_relevant_sentence(query)
        if not most_relevant_sentence:
            return "Je suis désolé, je n'ai pas trouvé de réponse pertinente."
        return most_relevant_sentence
    except Exception as e:
        return f"Une erreur est survenue: {e}"

# Fonction pour la transcription vocale avec sélection de l'API et de la langue
def transcribe_speech(api_choice, language):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Parlez maintenant...")
        audio_text = r.listen(source)
        st.info("Transcription en cours...")

        try:
            if api_choice == "Google":
                text = r.recognize_google(audio_text, language=language)
            elif api_choice == "Sphinx":
                text = r.recognize_sphinx(audio_text)
            elif api_choice == "IBM Watson":
                api_key = st.text_input("Clé API IBM Watson:", type="password")
                url = st.text_input("URL du service IBM Watson:")
                if api_key and url:
                    text = r.recognize_ibm(audio_text, username="apikey", password=api_key, url=url, language=language)
                else:
                    text = "Veuillez fournir les informations d'authentification pour IBM Watson."
            else:
                text = "API de reconnaissance non supportée"
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas pu comprendre l'audio."
        except sr.RequestError as e:
            return f"Erreur de service API : {e}"

# Fonction principale de l'application
def main():
    st.title("Chatbot à Commande Vocale")
    st.write("Bonjour ! Posez votre question en texte ou par la voix.")

    # Choix de la méthode d'entrée
    input_method = st.radio("Choisissez la méthode d'entrée", ["Texte", "Vocal"])

    # Entrée textuelle
    if input_method == "Texte":
        question = st.text_input("Vous:")
        if st.button("Envoyer"):
            response = chatbot(question)
            st.write("Chatbot: " + response)

    # Entrée vocale
    elif input_method == "Vocal":
        st.write("Assurez-vous que votre microphone fonctionne correctement.")
        api_choice = st.sidebar.selectbox("Choisissez l'API de reconnaissance vocale", ["Google", "Sphinx", "IBM Watson"])
        language = st.sidebar.selectbox("Choisissez la langue", ["fr-FR", "en-US", "es-ES", "de-DE"])
        if st.button("Commencer l'enregistrement"):
            text = transcribe_speech(api_choice, language)
            st.write("Texte transcrit : ", text)
            response = chatbot(text)
            st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()
