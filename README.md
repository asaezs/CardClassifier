# 🃏 Card Suit Classifier (Deep Learning)

This is a Machine Learning (Deep Learning) project developed in a Jupyter Notebook. The objective is to train a Convolutional Neural Network (CNN) to classify images of playing cards, not by their rank, but by their **suit** (Hearts, Diamonds, Spades, Clubs) and a fifth class for **outliers** (Jokers, card backs, etc.).

The project uses a public dataset from Kaggle containing thousands of varied card images.

---

## ✨ Techniques and Features

* **Libraries:** TensorFlow (Keras), Scikit-learn, OpenCV, Pandas, and Matplotlib.
* **Transfer Learning:** The `MobileNetV2` model (pre-trained on ImageNet) is used as a frozen "base" for feature extraction.
* **Handling Class Imbalance:** The `outlier` class was significantly smaller (approx. 5%) than the others. A dual strategy was implemented to combat this:
    1.  **Data Augmentation:** Keras's `ImageDataGenerator` was used to apply random transformations (rotation, zoom, etc.) to the training set.
    2.  **Class Weights:** A higher weight was calculated for the `outlier` class using `sklearn.utils.class_weight` and passed to `model.fit()` to penalize errors on this class more heavily.
* **Training and Evaluation:** The model was trained using *callbacks* like `EarlyStopping` and `ModelCheckpoint`. The final results were analyzed using a **Classification Report** and a **Confusion Matrix** from Scikit-learn.

---

## 📋 Notebook Methodology

The project follows a standard data science workflow:

1.  **Step 0: Setup:** Install dependencies.
2.  **Step 1: Load and Explore (EDA):**
    * Scan the dataset and load the information into a Pandas DataFrame.
    * Create a new `suit` column (Hearts, Diamonds, Spades, Clubs, Outlier) from the original labels.
    * Visualize the class imbalance.
3.  **Step 2: Data Preprocessing:**
    * Create Keras `ImageDataGenerator`s for training, validation, and testing.
    * Apply an aggressive Data Augmentation recipe to the training generator.
    * Calculate the `class_weights` for training.
4.  **Step 3: Model Building:**
    * Load the frozen `MobileNetV2` base (`include_top=False`).
    * Add a custom classification "head" (`GlobalAveragePooling2D`, `Dense(128)`, `Dropout(0.5)`, `Dense(5, 'softmax')`).
    * Compile the model with the `Adam` optimizer.
5.  **Step 4: Training:**
    * Run `model.fit()`, passing in the generators and `class_weights`.
    * Use *callbacks* (like `EarlyStopping`).
6.  **Step 5: Evaluation:**
    * Evaluate the final accuracy against the `test_generator`.
    * Generate and visualize the Classification Report and Confusion Matrix to analyze where the model fails (e.g., confusion between Spades and Clubs).

---

## 📈 Results

The final model achieved a test accuracy of approximately **80%**. The confusion matrix analysis showed high effectiveness in distinguishing colors (Hearts vs. Diamonds) but, as expected, more difficulty with the `outlier` class (which improved significantly thanks to class weights) and in the confusion between the two black suits, Spades and Clubs.

---

## ⚙️ Setup and Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (On Linux/macOS)
    .\venv\Scripts\activate   # (On Windows)
    ```

3.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib opencv-python scikit-learn jupyterlab
    ```

4.  **Run Jupyter Lab:**
    ```bash
    jupyter lab
    ```

5.  Open the `.ipynb` file and run the cells.
