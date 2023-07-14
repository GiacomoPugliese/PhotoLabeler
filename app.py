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
import shutil
import datetime
import re
from zipfile import ZipFile, ZIP_STORED

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
    directory = os.path.join(os.getcwd(), collection_id, 'training_imgs', person_name)
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
    

st.title("Leadership Initiatives Photo Labeler")

with st.expander("Click to view full directions for this site"):
    st.subheader("Configure Training Data")
    st.write("- Either create or sign in to your program by entering a program ID (** a program ID is needed for all parts of the site **)")
    st.write("- Create student profiles and upload solo images of them to train the AI (** names must be in format 'FIRST_LAST' **)")
    st.write("- Delete student profile to clear their training data")
    st.subheader("Interns in System")
    st.write("- Displays all of the interns currently in your program's AI")
    st.subheader("Detect Interns in Photos")
    st.write("- Upload one or more zip folders containing pictures of the interns within the program")
    st.write("- Click 'Start Processing' and allow the AI to sort the images into individual student folders (may take a while)")
    st.write("- Download the zip folder with all of the sorted images to your computer")
    st.subheader("Renaming tool")
    st.write("- Upload a zip folder of images of program's students at a particular location")
    st.write("- Choose the custom file ending for that location (i.e. '_Jumpstart_Group_1' for a file you want named 'First_Last_2023_Jumpstart_Group_1')")
    st.write("- Click 'Start Renaming' and download a zip folder of the automatically renamed image files")
# Add a person or image, or delete a person
st.header('Configure Training Data')

collection_id = st.text_input("Enter your program ID", "")
if collection_id == '':
    collection_id = 'your-default-collection-id'
create_collection(collection_id)

# Create a directory named after the collection
os.makedirs(collection_id, exist_ok=True)

person_name = st.text_input("Enter the intern's name")
person_image = st.file_uploader('Upload a solo image of the intern', type=['jpg', 'png'])

if st.button('Add image'):
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
    else:
        if person_name and person_image and collection_id:
            # Save the image locally
            file_path = save_file_locally(person_image, person_name)
            # Upload the image to S3 bucket
            with open(file_path, "rb") as f:
                upload_success = upload_file_to_s3(f, 'giacomo-aws-bucket', person_name)
            if upload_success:
                # Check if person already exists
                if person_name not in list_faces_in_collection(collection_id):
                    add_faces_to_collection('giacomo-aws-bucket', person_name, collection_id, person_name)
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
            st.write('Please enter a name, upload an image, and provide a collection ID')

if st.button('Delete intern'):
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
    else:
        if person_name and collection_id:
            # Find the faceId of the person to delete
            face_id = next((face['FaceId'] for face in client.list_faces(CollectionId=collection_id)['Faces'] 
                            if face['ExternalImageId'] == person_name), None)
            if face_id:
                delete_face_from_collection(collection_id, face_id)
                st.write(f'Person {person_name} deleted successfully')
            else:
                st.write(f'Person {person_name} not found')
        else:
            st.write('Please enter a name and collection ID to delete')

# Display the list of person names
st.header('Interns in System')
person_names = list_faces_in_collection(collection_id)
st.write(', '.join(person_names))
if len(person_names) == 0:
    'No interns added yet.'
st.button("Refresh page")

# Get the current year
current_year = datetime.datetime.now().year

st.header('Detect Interns in Photos')
uploaded_files = st.file_uploader('Upload zip files containing photos', type=['zip'], accept_multiple_files=True)
start_processing = st.button('Start Processing')

if start_processing and uploaded_files:
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
    else:
        # Create a temporary directory to hold the extracted files
        if not os.path.exists('temp'):
            os.makedirs('temp')

        total_files = 0
        for uploaded_file in uploaded_files:
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
                zip_ref.extractall('temp')
                total_files += len(zip_ref.namelist())

        # Create a dictionary to hold the zip files for each person
        zip_files = {person: zipfile.ZipFile(f'{person}.zip', 'w') for person in st.session_state['person_names']}

        progress_report = st.empty()  # Create a placeholder for the progress report

        with st.spinner("Labeling images.."):
            for i, file_name in enumerate(os.listdir('temp'), start=1):
                try:
                    progress_report.text(f"Labeling progress: ({i}/{total_files})")  # Update the text in the placeholder
                    with open(os.path.join('temp', file_name), 'rb') as file:
                        img = resize_image(file, 1000)  # Adjust the width as needed
                        byte_arr = io.BytesIO()
                        img.save(byte_arr, format='JPEG')  # Or format='PNG' if your images are PNG
                        byte_img = byte_arr.getvalue()
                        detected_persons = find_matching_faces(byte_img, collection_id)
                        print(f'{file_name}: {", ".join(detected_persons)}')

                        # Add the image to the zip file of each detected person
                        for person in detected_persons:
                            if person not in zip_files:
                                zip_files[person] = zipfile.ZipFile(f'{collection_id}/{person}.zip', 'w')
                            file.seek(0)  # Seek back to the start of the file

                            # Generate the new file name
                            new_file_name = f"{person}_{current_year}_{file_name}"

                            zip_files[person].writestr(new_file_name, file.read())
                except Exception as e:
                    print(f"{file_name} threw an error: {e}")
                    continue

        # Create a zip file to hold all person zip files
        all_persons_zip = zipfile.ZipFile(f'{collection_id}/all_persons.zip', 'w')

        total_persons = len(zip_files.items())
        zip_progress_report = st.empty()  # Create a placeholder for the zip progress report

        with st.spinner("Generating zip folders..."):
            st.subheader("Download Link:")
            # Close the individual zip files, add them to the all persons zip file and provide a download link
            for i, (person, zip_file) in enumerate(zip_files.items(), start=1):
                zip_file.close()
                zip_progress_report.text(f"Zip folder progress: ({i}/{total_persons})")

                # Add the individual zip file to the all persons zip file
                with open(f'{collection_id}/{person}.zip', 'rb') as f:
                    all_persons_zip.writestr(f'{person}.zip', f.read())

        # Close the all persons zip file
        all_persons_zip.close()

        st.session_state['download_zip_created'] = True  

        # Cleanup the temp directory
        shutil.rmtree('temp')

# Replace the markdown link with a download button
if 'download_zip_created' in st.session_state and st.session_state['download_zip_created']:  
    with open(f'{collection_id}/all_persons.zip', 'rb') as f:
        st.download_button(
            label="Download all images",
            data=f.read(),
            file_name='all_persons.zip',
            mime='application/zip'
        )

st.header('Naming Tool')
uploaded_zip_file = st.file_uploader('Upload a zip file to rename files', type=['zip'])
file_name_ending = st.text_input('Enter your custom file name ending')
start_renaming = st.button('Start Renaming')

if start_renaming and uploaded_zip_file:
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
    else:
        try:
            # Create a temporary directory to hold the extracted files
            if not os.path.exists('temp'):
                os.makedirs('temp')

            # Extract the zip file
            with ZipFile(uploaded_zip_file, 'r') as zip_ref:
                zip_ref.extractall('temp')

            # Create a new zip file to hold the renamed files
            renamed_files_zip = ZipFile(f'{collection_id}/renamed_files.zip', 'w', compression=ZIP_STORED)
        
            for file_name in os.listdir('temp'):
                print(file_name)
                if '_2023_' in file_name:
                    # Remove all characters past '_2023_' in the file name
                    new_file_name = re.sub(r'_2023_.*', '', file_name)

                    # Append the custom file name ending
                    new_file_name += f'_2023_{file_name_ending}'

                    # Add the file to the new zip file with the new name
                    renamed_files_zip.write(os.path.join('temp', file_name), arcname=new_file_name)

            # Close the zip file
            renamed_files_zip.close()

            # Cleanup the temp directory
            shutil.rmtree('temp')

            st.session_state['renamed_files_created'] = True
        except:
            st.error("Please make sure you've uploaded a zip folder of images!")

# Provide a download button for the renamed files zip
if 'renamed_files_created' in st.session_state and st.session_state['renamed_files_created']:  
    with open(f'{collection_id}/renamed_files.zip', 'rb') as f:
        st.download_button(
            label="Download renamed files",
            data=f.read(),
            file_name='renamed_files.zip',
            mime='application/zip'
        )
