# Zeki Müren Turkish Song Generator

This project generates new Turkish song lyrics inspired by the works of Zeki Müren using deep learning and natural language processing techniques.

## Features

- Generates original Turkish lyrics based on Zeki Müren's songs.
- Includes data cleaning, preprocessing, and text generation scripts.
- Supports creative lyric generation with temperature and sampling adjustments.

## Usage

1. **Data Preparation:**  
   Run `data_preparation.py` to clean and preprocess the lyrics dataset. This script will generate the necessary files for model training.

2. **Model Training:**  
   *(Model training code should be added separately. This repository currently includes data preparation and lyric generation.)*

3. **Lyric Generation:**  
   Use `song_generator.py` to generate new lyrics with a trained model. Example usage:
   ```python
   seed_text = "Sevgilim"
   print(generate_song(seed_text, next_words=100))
   ```

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## File Descriptions

- `data_preparation.py`: Cleans and prepares the dataset for training.
- `song_generator.py`: Generates new lyrics using the trained model.
- `zeki.csv`: The dataset of Zeki Müren's song lyrics.
- `tokenizer.pickle`, `max_sequence_len.npy`: Helper files for the model.

## Contribution

Feel free to open an issue or submit a pull request for improvements or suggestions.

## License

This project is licensed under the MIT License.

---

**Note:** This project is for educational and research purposes only and is intended as a tribute to Zeki Müren’s legacy.
