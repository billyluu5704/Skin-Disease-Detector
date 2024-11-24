# streamlitapp.py
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from run_test import recreate_model, diseases  # Import recreate_model and disease labels

# Load the pre-trained model
model_file = 'train_model.keras'
model = load_model(model_file, custom_objects={'Resnet': recreate_model()})

def load_image(uploaded_file):
    """Load and preprocess the image for prediction."""
    img = Image.open(uploaded_file).resize((224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img -= [123.68, 116.779, 103.939]
    return img

def predict_disease(img):
    """Predict the skin disease from the image."""
    pred = model.predict(img)
    result = np.argmax(pred, axis=1)[0]
    return result

# Streamlit layout
def main():
    st.title("Skin Disease Diagnoser")
    st.markdown("### Upload an image to get started! 📷")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Run model prediction when the button is clicked
        if st.button("Run Model"):
            img = load_image(uploaded_file)
            result = predict_disease(img)

            # Display the result based on prediction
            st.markdown("### Model Prediction:")
            if 0 <= result < len(diseases):
                st.write(f"You may have {diseases[result]}.")

                # Custom message for each disease (based on previous elif cases)
                if result == 0:
                    st.write("High chance of Eczema. Please consult a doctor and consider cortocosteroids and treatment cream.")
                elif result == 1:
                    st.write("This may be Melanoma (skin cancer). Visit the hospital immediately for examination.")
                elif result == 2:
                    st.write("Possible Atopic Dermatitis. Try moisturizing; if it persists, consider medicated creams.")
                elif result == 3:
                    st.write("Possible Basel Cell Carcinoma. Consider seeing a doctor for Hydroquinone Cream.")
                elif result == 4:
                    st.write("Likely Melanocytic Nevi. Imiquimod 5% Cream is recommended.")
                elif result == 5:
                    st.write("Benign Keratosis-like Lesions. Generally harmless but consult a healthcare provider if concerned.")
                elif result == 6:
                    st.write("Psoriasis Lichen Planus. Corticosteroid creams, Antihistamines, and Phototherapy may help.")
                elif result == 7:
                    st.write("Seborrheic Keratoses. Consider Cryotherapy or Laser Therapy.")
                elif result == 8:
                    st.write("Tinea Ringworm or Candidiasis. Consider Clotrimazole or Miconazole creams.")
                elif result == 9:
                    st.write("Warts Molluscum. Consider Imiquimod cream; consult a healthcare provider if it worsens.")
            else:
                st.write("No recognizable disease prediction. Please consult a doctor for further advice.")

# Run the app
if __name__ == "__main__":
    main()