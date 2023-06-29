"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from preprocess import clean_text

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
raw_test = pd.read_csv("resources/test_with_no_labels.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Data Alchemist Tweet Classifer")
	#st.subheader("Climate change tweet classification")
	#adding our logo
	
	#st.sidebar.image('resources/imgs/app_logo.JPG', width=100)
	#st.image('resources/imgs/app_logo.JPG', width=100)
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home","How it works","Make a Prediction", "Unprocessed Data","About Us","Contact Us"]
	selection = st.sidebar.selectbox("Choose Menu Option", options)
	#Building the Home Page
	if selection == "Home":
		st.title("Data Alchemist Tweet Classifer")
		st.info("Home page")
		st.image('resources/imgs/app_logo.JPG' )
		st.subheader("Tweet Categorizer")
		st.markdown('<div style="text-align: justify;">Data Alchemist is a company dedicated to helping businesses reduce their environmental impact. As data scientists, we provide accurate and reliable solutions that allow companies to access a wide range of consumer sentiment data across different demographics and locations. By leveraging this information, businesses can gain valuable insights to inform their future marketing strategies. Our robust solutions empower companies to make more sustainable decisions and drive impactful marketing initiatives.</div>', unsafe_allow_html=True)
		
	# Building out the "Information" page
	if selection == "Unprocessed Data":
		st.title("Unprocessed Data")
		# You can read a markdown file from supporting resources folder
		st.markdown("Unprocessed Data")

		st.subheader("Raw Twitter data with Labels: For Model Training")
		if st.checkbox('Show/Hide Train Data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		st.subheader("Raw Twitter data without Labels: For Model Testing")
		st.info("You can use this tweats to make sample prediction")
		if st.checkbox('Show/Hide Test Data'): # data is hidden if box is unchecked
			st.write(raw_test[['message']]) # will write the df to the page
	#About Us
	if selection == "About Us":
		st.title("About Us")
		st.subheader("Mission")
		#st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True)
		st.markdown('<div style="text-align: justify;">At Data Alchemist, our mission is to harness the power of data to drive positive change and innovation. We strive to empower businesses and organizations with transformative insights, enabling them to make informed decisions, optimize operations, and create sustainable solutions. By unlocking the hidden value within data, we aim to catalyze progress, foster growth, and contribute to a better future for individuals, businesses, and society as a whole.</div>', unsafe_allow_html=True)
		st.subheader("Vision")
		st.markdown('<div style="text-align: justify;">Our vision is to be at the forefront of the data revolution, leading the way in transforming data into actionable intelligence. We envision a world where data is harnessed ethically and responsibly to address complex challenges, improve decision-making processes, and drive meaningful impact. Through our cutting-edge technologies, expertise in data science, and unwavering commitment to innovation, we aspire to be the trusted partner for organizations seeking to unlock the full potential of their data and embrace data-driven transformation.</div>', unsafe_allow_html=True)
		st.image('resources/imgs/app_logo.JPG' )
	#Guide-How it Works
	if selection == "How it works":
		st.title("How it works")
		#Overview
		st.subheader("Overview")
		st.markdown('<div style="text-align: justify;">Our application utilizes a sophisticated tweet classification model to categorize tweets based on their sentiment towards climate change. Additionally, it goes a step further by identifying whether the tweet expresses the user&#39s stance on man-made climate change. The underlying model employs state-of-the-art natural language processing techniques, including sentiment analysis and contextual understanding, to accurately assess the sentiment and stance portrayed in each tweet. This provides a simplified yet comprehensive description of each tweet&#39s climate change sentiment and indication of the user&#39s stance.</div>', unsafe_allow_html=True)

		#Data Eng
		st.subheader("Data Cleaning and Feature Engineering")
		st.markdown('<div style="text-align: justify;">Prior to performing tweet classification, a crucial step involves data cleaning, which encompasses the identification, correction, or removal of errors, inconsistencies, and inaccuracies within the dataset. This process employs diverse techniques and operations aimed at enhancing the overall quality and reliability of the data. In the context of our task, the objective is to classify tweets based on the presence or absence of specific words or word combinations, focusing solely on those that effectively convey meaning.</div>', unsafe_allow_html=True)

		#Modelling
		st.subheader("Modelling")
		st.markdown('<div style="text-align: justify;">We conducted comprehensive evaluations of diverse machine learning models and meticulously optimized them to identify the best performing model. The evaluation process revolved around the model&#39s capability to accurately classify tweets, utilizing an existing dataset as the benchmark. </div>', unsafe_allow_html=True)
		
		#Prediction
		st.subheader("Tweet Sentiment Prediction")
		st.markdown('<div style="text-align: justify;">In the final stage of our data science pipeline, we employed our optimized model to predict the sentiment of a given tweet by leveraging its textual content. Utilizing the trained model&#39s learned patterns and features, we applied it to new, unseen tweets to infer their sentiment.</div>', unsafe_allow_html=True)
		
	#Contact Us
	if selection == "Contact Us":
		st.title("Contact Us")
		st.write("Please feel free to reach out to us using the contact details below:")
		# Contact details
		st.write("Email: info@dataalchemy.com")
		st.write("Phone: (021) 456 7890")
		st.write("Address: 123 Albany St., Cape Town, Mailing Address Western cape, South Africa")
	# Building out the predication page
	if selection == "Make a Prediction":
		st.title("Data Alchemist Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("Below you required to Select the Model, Enter the tweet and press Classify")

		#adding different Models
		options_m = ["Logistic Regression", "Random Forest", "Multinomial Naive Bayes", "K-Nearest Nabour","Support Vector Machine" ]
		selection_m = st.selectbox("Choose Model", options_m)
		#getting path to the model files
		if selection_m == "Logistic Regression":
			#path_m = "resources/Logistic_regression.pkl"
			path_m = "resources/lr_model.pkl"
		if selection_m == "Random Forest":
			path_m = "resources/rf_model.pkl"
		if selection_m == "Multinomial Naive Bayes":
			path_m = "resources/mnb_model.pkl"
		if selection_m == "Support Vector Machine":
			path_m = "resources/svm_model.pkl"
		if selection_m == "K-Nearest Nabour":
			path_m = "resources/knn_model.pkl"
		# Creating a text box for user input
		input_text = st.text_area("Enter Tweet Below:","")
		tweet_text = clean_text(input_text)

		
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice


			predictor = joblib.load(open(os.path.join(path_m),"rb"))



			prediction = predictor.predict(vect_text)
			# add description about the tweet predction----
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			key_desc = {"-1": "Anti: This means that the tweet does not believe in man-made climate change Variable definitions",
	       "0":"Neutral: This means that the tweet neither supports nor refutes the belief of man-made climate change",
	       "1":"Pro: This means that the tweet supports the belief of man-made climate change",
	       "2":"News: This means that the tweet links to factual news about climate change"
		   }
			pred_desc = prediction[0]
			st.success("Tweet is Classified as: {}".format(key_desc[str(pred_desc)])+"\n\n Prediction by: "+selection_m+" Model")
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
