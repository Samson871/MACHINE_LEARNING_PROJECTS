# Twitter Sentiment Analysis

This project is a web application for analyzing the sentiment of tweets using a Logistic Regression model. The application is built using Flask and provides a user-friendly interface for inputting tweet text and receiving sentiment predictions.

## Project Structure

- `.vscode/settings.json`: Configuration for Visual Studio Code.
- `app.py`: Main Flask application file.
- `dataset/`: Directory for storing datasets.
- `model/`: Directory for storing the trained model and vectorizer.
- `notebook/twitter_sentiment_analysis_using_logrgrsn.py`: Jupyter notebook for training the model.
- `requirements.txt`: List of dependencies.
- `static/styles.css`: CSS file for styling the web application.
- `templates/index.html`: HTML template for the web application.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download NLTK stopwords:
    ```sh
    python -m nltk.downloader stopwords
    ```

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Enter the tweet text in the provided textarea and click "Predict Sentiment" to get the sentiment prediction.

## Model Training

The model is trained using a Jupyter notebook located at [twitter_sentiment_analysis_using_logrgrsn.py](http://_vscodecontentref_/8). The notebook includes steps for data preprocessing, model training, and saving the trained model and vectorizer.

## Files

- [app.py](http://_vscodecontentref_/9): Contains the Flask application code, including routes and sentiment prediction logic.
- [trained_model.sav](http://_vscodecontentref_/10): Serialized Logistic Regression model.
- [vectorizer.pickle](http://_vscodecontentref_/11): Serialized TfidfVectorizer.
- [styles.css](http://_vscodecontentref_/12): CSS styles for the web application.
- [index.html](http://_vscodecontentref_/13): HTML template for the web application.
- [requirements.txt](http://_vscodecontentref_/14): List of dependencies required for the project.
- [twitter_sentiment_analysis_using_logrgrsn.py](http://_vscodecontentref_/15): Jupyter notebook for training the model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Bootstrap](https://getbootstrap.com/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact [yourname@example.com](mailto:yourname@example.com).