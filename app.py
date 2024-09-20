import streamlit as st
from PIL import Image
import os
import httpx
import auth
from gradio_client import Client, handle_file

# ------- PAGE CONFIGURATION -------
st.set_page_config(layout="wide", page_title="Virtual Try-On")

# ------- USER AUTHENTICATION -------
auth.create_user_table()

# ------- FUNCTIONS -------
def show_signup_page():
    st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #ffffff;'>
            <h1 style='color: #1e90ff; font-family: Montserrat, sans-serif;'>Join the Style Revolution!</h1>
            <p style='font-size: 16px; color: #777;'>Create an account to start your fashion journey</p>
        </div>
        <hr style='margin-top: 20px;'>
    """, unsafe_allow_html=True)
    
    st.subheader("Create New Account")
    name = st.text_input("Name", key='signup_name')
    username = st.text_input("Username", key='signup_username')
    password = st.text_input("Password", type="password", key='signup_password')
    confirm_password = st.text_input("Confirm Password", type="password", key='signup_confirm_password')
    
    if st.button("Sign Up", key='signup_button'):
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            error = auth.add_user(name, username, password)
            if error:
                st.error(error)
            else:
                st.success("Account created successfully! You are now registered.")
                st.session_state['registered'] = True

def show_login_page():
    st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f9f9f9;'>
            <h1 style='color: #ff6347; font-family: Open Sans, sans-serif;'>VTON By Dofday</h1>
            <p style='font-size: 16px; color: #777;'>Your ultimate virtual fashion try-on experience</p>
        </div>
        <hr style='margin-top: 20px;'>
    """, unsafe_allow_html=True)
    
    st.subheader("Login")
    username = st.text_input("Username", key='login_username')
    password = st.text_input("Password", type="password", key='login_password')
    
    if st.button("Login", key='login_button'):
        user = auth.authenticate_user(username, password)
        if user:
            st.session_state['user'] = user
            st.rerun()
        else:
            st.error("Username/password is incorrect")

if 'user' not in st.session_state:
    page = st.sidebar.selectbox("Select Page", ["Login", "Sign Up"], key='select_page')
    if page == "Login":
        show_login_page()
    else:
        show_signup_page()
else:
    user = st.session_state['user']
    st.sidebar.title(f"Welcome {user[1]}")
    if st.sidebar.button("Logout", key='logout_button'):
        del st.session_state['user']
        st.rerun()

    # ------- CONSTANTS -------
    CLOTH_TYPES = ["upper", "lower", "overall"]
    SHOW_TYPES = ["input & mask & result", "input & result"]
    IMAGE_DIMENSION = 300
    IMAGE_CONTAINER_WIDTH = IMAGE_DIMENSION + 50

    # Columns for displaying images
    col1, col2 = st.columns(2)

    # ------- WARDROBE SECTION -------
    wardrobe_dir = "wardrobe"
    os.makedirs(wardrobe_dir, exist_ok=True)
    wardrobe_files = os.listdir(wardrobe_dir)
    wardrobe_images = [os.path.join(wardrobe_dir, file) for file in wardrobe_files if file.endswith(('.png', '.jpeg', '.jpg'))]
    
    st.sidebar.title("Wardrobe")
    selected_wardrobe_image = st.sidebar.selectbox("Select a garment", wardrobe_images, key='select_garment')
    uploaded_file = st.sidebar.file_uploader("Or upload a new garment", type=['png', 'jpg', 'jpeg'], key='upload_garment')
    if uploaded_file:
        with open(os.path.join(wardrobe_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("Uploaded successfully!")

    # ------- PARAMETERS UI -------
    cloth_type = st.sidebar.selectbox("Cloth Type", CLOTH_TYPES, key='select_cloth_type')
    show_type = st.sidebar.selectbox("Show Type", SHOW_TYPES, key='select_show_type')
    num_inference_steps = st.sidebar.slider("Number of Inference Steps", min_value=50, max_value=200, value=100, key='slider_inference_steps')
    guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=10.0, value=2.5, key='slider_guidance_scale')
    seed = st.sidebar.slider("Seed", min_value=1, max_value=100, value=42, key='slider_seed')

    # ------- TRY ON FUNCTION -------
    def try_on(model_image_path, garment_image_path):
        try:
            client = Client("http://120.76.142.206:8888/")
            result = client.predict(
                 person_image={
                    "background":handle_file(model_image_path),
                    "layers":[handle_file('http://120.76.142.206:8888/file=/data1/chongzheng_p23/tmp/gradio/b2072ce2d50d9c0cef8f92bdd67f1eeeec37962c/layer_0.png')],
                    "composite":handle_file(model_image_path)},
                cloth_image=handle_file(garment_image_path),
                cloth_type=cloth_type,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                show_type=show_type,
                api_name="/submit_function"
            )
            return result
        except httpx.HTTPError as e:
            st.error("Failed to connect to the server: " + str(e))
            return None

    # ------- IMAGE UPLOAD AND DISPLAY -------
    model_image_file = st.sidebar.file_uploader("Select Model Image", type=["jpeg", "jpg", "png"], key='select_model_image')
    if model_image_file:
        model_image_path = os.path.join("temp_uploads", model_image_file.name)
        with open(model_image_path, "wb") as f:
            f.write(model_image_file.getbuffer())
        model_image = Image.open(model_image_path)
        model_image = model_image.resize((IMAGE_DIMENSION, IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        col1.image(model_image, caption="Model Image", width=IMAGE_CONTAINER_WIDTH)

    if selected_wardrobe_image:
        garment_image = Image.open(selected_wardrobe_image)
        garment_image = garment_image.resize((IMAGE_DIMENSION, IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        col2.image(garment_image, caption="Garment Image", width=IMAGE_CONTAINER_WIDTH)

    if st.sidebar.button("Try On", key='try_on_button'):
        if not model_image_path or not selected_wardrobe_image:
            st.error("Please select both model and garment images.")
        else:
            result = try_on(model_image_path, selected_wardrobe_image)
            if result:
                st.image(result, caption="Output Image", width=IMAGE_CONTAINER_WIDTH)
