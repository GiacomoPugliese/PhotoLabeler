# Streamlit App to use Amazon Rekognition API

import streamlit as st
import boto3
import os
from PIL import Image
import tempfile
from botocore.exceptions import ClientError
import logging
import zipfile
import io
from io import BytesIO
import base64

logging.basicConfig(level=logging.INFO)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title='SmartLabel',
    page_icon='camera'
)   
hide_streamlit_style = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """ 
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set AWS details (replace with your own details)
AWS_REGION_NAME = 'us-east-2'
AWS_ACCESS_KEY = 'AKIARK3QQWNWXGIGOFOH'
AWS_SECRET_KEY = 'ClAUaloRIp3ebj9atw07u/o3joULLY41ghDiDc2a'

# Initialize the S3 client
s3 = boto3.client('s3',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Initialize boto3 client for AWS Rekognition
client = boto3.client('rekognition',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def save_file_locally(file, person_name):
    """
    Save the uploaded file to a local directory named after the person.

    :param file: File to save
    :param person_name: Name of the person. Used to create a directory for the person.
    :return: The path to the saved file
    """
    directory = os.path.join(os.getcwd(), 'training_imgs', person_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path


def resize_image(image, basewidth):
    img = Image.open(image)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def delete_collection(collection_id):
    try:
        response = client.delete_collection(CollectionId=collection_id)
        st.write('Deleted collection:', collection_id)
    except ClientError as e:
        logging.error(e)
        raise e
    
def create_collection(collection_id):
    try:
        response = client.create_collection(CollectionId=collection_id)
        pass
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
            pass
        else:
            logging.error(e)
            raise e
        
def upload_file_to_s3(file, bucket, object_name):
    """
    Upload a file to an S3 bucket

    :param file: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    try:
        s3.upload_fileobj(file, bucket, object_name)
    except ClientError as e:
        logging.error('Error uploading file to S3: %s', e)
        return False
    logging.info('File uploaded successfully to bucket %s with key %s', bucket, object_name)
    return True


# Define function to add faces to the collection
def add_faces_to_collection(bucket, photo, collection_id, external_image_id):
    response = client.index_faces(
        CollectionId=collection_id,
        Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        ExternalImageId=external_image_id,
        MaxFaces=5,
        QualityFilter="AUTO",
        DetectionAttributes=['ALL']
    )

    return response['FaceRecords']

def find_matching_faces(photo, collection_id):
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(photo))

    # Attempt to detect faces in the photo
    faces = client.detect_faces(Image={'Bytes': photo})
    # print(f"Number of faces detected by detect_faces: {len(faces['FaceDetails'])}")
    
    # For each face, search for matches in the collection
    face_matches = []
    for i, face in enumerate(faces['FaceDetails']):
        bbox = face['BoundingBox']
        # Convert bbox to the format required by search_faces_by_image
        width, height = image.width, image.height
        left = int(bbox['Left']*width)
        top = int(bbox['Top']*height)
        face_width = int(bbox['Width']*width)
        face_height = int(bbox['Height']*height)

        # Add a buffer to the bounding box coordinates to make sure the entire face is included
        buffer = 20  # adjust this value as needed
        left = max(0, left - buffer)
        top = max(0, top - buffer)
        right = min(width, left + face_width + buffer)
        bottom = min(height, top + face_height + buffer)

        face_crop = image.crop((left, top, right, bottom))
        bytes_io = io.BytesIO()
        face_crop.save(bytes_io, format='JPEG')
        face_bytes = bytes_io.getvalue()

        try:
            response = client.search_faces_by_image(
                CollectionId=collection_id,
                Image={'Bytes': face_bytes},
                FaceMatchThreshold=70,
                MaxFaces=5
            )
            face_matches.extend([match['Face']['ExternalImageId'] for match in response['FaceMatches']])
            # print(f"Face {i+1}: matches found")
        except Exception as e:
            pass
            # print(f"Face {i+1}: error occurred - {str(e)}")

    # print(f"Number of face matches found: {len(face_matches)}")
    return face_matches




def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    return href

# Define function to list faces in the collection
def list_faces_in_collection(collection_id):
    response = client.list_faces(CollectionId=collection_id)
    return [face['ExternalImageId'] for face in response['Faces']]

# Define function to delete a face from the collection
def delete_face_from_collection(collection_id, face_id):
    response = client.delete_faces(CollectionId=collection_id, FaceIds=[face_id])
    return response['DeletedFaces']

# Define function to add a training image to an existing person
def add_training_image_to_person(collection_id, person_name, image):
    # Upload the image to S3 bucket
    object_name = f'{person_name}_{len(list_faces_in_collection(collection_id)) + 1}.jpg'
    upload_success = upload_file_to_s3(image, 'giacomo-aws-bucket', object_name)
    if upload_success:
        add_faces_to_collection('giacomo-aws-bucket', object_name, collection_id, object_name)

if 'person_names' not in st.session_state:
    st.session_state['person_names'] = []
    st.session_state['last_uploaded_file'] = None
    st.session_state['download_zip_created'] = False
    # delete_collection('your-collection-id')
    
# Create the collection at the start of your app
create_collection('your-collection-id')

st.title("AI Photo Labeler")
# Add a person or image, or delete a person
st.header('Configure Training Data')
person_name = st.text_input("Enter the intern's name")
person_image = st.file_uploader('Upload a solo image of the intern', type=['jpg', 'png'])

if st.button('Add image'):
    if person_name and person_image:
        # Save the image locally
        file_path = save_file_locally(person_image, person_name)
        # Upload the image to S3 bucket
        with open(file_path, "rb") as f:
            upload_success = upload_file_to_s3(f, 'giacomo-aws-bucket', person_name)
        if upload_success:
            # Check if person already exists
            if person_name not in list_faces_in_collection('your-collection-id'):
                add_faces_to_collection('giacomo-aws-bucket', person_name, 'your-collection-id', person_name)
                st.session_state['person_names'].append(person_name)
                st.write('Person added successfully')
            else:
                # If person already exists, just add the image to the person's existing images in the collection
                st.write('Image added to existing person')
            # Clear the input fields
            person_name = None
            person_image = None
        else:
            st.write('Failed to upload image')
    else:
        st.write('Please enter a name and upload an image')

if st.button('Delete intern'):
    if person_name:
        # Find the faceId of the person to delete
        face_id = next((face['FaceId'] for face in client.list_faces(CollectionId='your-collection-id')['Faces'] 
                        if face['ExternalImageId'] == person_name), None)
        if face_id:
            delete_face_from_collection('your-collection-id', face_id)
            st.write(f'Person {person_name} deleted successfully')
        else:
            st.write(f'Person {person_name} not found')
    else:
        st.write('Please enter a name to delete')

# Display the list of person names
st.header('Interns in System')
person_names = list_faces_in_collection('your-collection-id')
st.write(', '.join(person_names))
st.button("Refresh page")

st.header('Detect Interns in Photos')
uploaded_file = st.file_uploader('Upload a zip file containing photos', type=['zip'])

if uploaded_file is not None and uploaded_file != st.session_state['last_uploaded_file']:
    st.session_state['last_uploaded_file'] = uploaded_file

    # Create a dictionary to hold the zip files for each person
    zip_files = {person: zipfile.ZipFile(f'{person}.zip', 'w') for person in st.session_state['person_names']}

    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        total_files = len(zip_ref.namelist())
        progress_report = st.empty()  # Create a placeholder for the progress report
        with st.spinner("Labeling images.."):
            for i, file_name in enumerate(zip_ref.namelist(), start=1):
                try:
                    progress_report.text(f"Labeling progress: ({i}/{total_files})")  # Update the text in the placeholder
                    with zip_ref.open(file_name) as file:
                        img = resize_image(file, 1000)  # Adjust the width as needed
                        byte_arr = io.BytesIO()
                        img.save(byte_arr, format='JPEG')  # Or format='PNG' if your images are PNG
                        byte_img = byte_arr.getvalue()
                        detected_persons = find_matching_faces(byte_img, 'your-collection-id')
                        print(f'{file_name}: {", ".join(detected_persons)}')

                        # Add the image to the zip file of each detected person
                        for person in detected_persons:
                            if person not in zip_files:
                                zip_files[person] = zipfile.ZipFile(f'{person}.zip', 'w')
                            file.seek(0)  # Seek back to the start of the file
                            zip_files[person].writestr(file_name, file.read())
                except Exception as e:
                    print(f"{file_name} threw an error: {e}")
                    continue

    # Create a zip file to hold all person zip files
    all_persons_zip = zipfile.ZipFile('all_persons.zip', 'w')

    total_persons = len(zip_files.items())
    zip_progress_report = st.empty()  # Create a placeholder for the zip progress report
    with st.spinner("Generating zip folders..."):
        st.subheader("Download Link:")
        # Close the individual zip files, add them to the all persons zip file and provide a download link
        for i, (person, zip_file) in enumerate(zip_files.items(), start=1):
            zip_file.close()
            zip_progress_report.text(f"Zip folder progress: ({i}/{total_persons})")

            # Add the individual zip file to the all persons zip file
            with open(f'{person}.zip', 'rb') as f:
                all_persons_zip.writestr(f'{person}.zip', f.read())

    # Close the all persons zip file
    all_persons_zip.close()

    st.session_state['download_zip_created'] = True  

# Replace the markdown link with a download button
if st.session_state['download_zip_created']:  
    with open('all_persons.zip', 'rb') as f:
        st.download_button(
            label="Download all images",
            data=f.read(),
            file_name='all_persons.zip',
            mime='application/zip'
        )
