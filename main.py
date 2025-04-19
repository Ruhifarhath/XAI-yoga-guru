import streamlit as st
from utils.rag import get_rag_chain
from utils.groq_rag import answer_with_groq
from model.explainer import generate_pose_prediction_and_explanation
import base64

def add_bg_from_local(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("bg2.jpg")  # 👈 Call it right here


      
st.title("AI-Powered Yoga Guidance")


# Initialize session state variables
if "selected_pose" not in st.session_state:
    st.session_state.selected_pose = "Child’s Pose"  # Set a default pose

    if "chat_step" not in st.session_state:
        st.session_state.chat_step = 0

# Initialize session state
if "chat_step" not in st.session_state:
    st.session_state.chat_step = 0
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

# Initialize safe and unsafe poses in session state
if "safe_poses" not in st.session_state:
    st.session_state.safe_poses = [
        "Child’s Pose", "Bound Angle Pose", "Cat-Cow Pose",
        "Downward Facing Dog Pose", "Mountain Pose", "Angle Pose"
    ]
if "unsafe_poses" not in st.session_state:
    st.session_state.unsafe_poses = ["Example Unsafe Pose 1", "Example Unsafe Pose 2"]

# Step 0: Welcome Message + Food & Water Precautions
if st.session_state.chat_step == 0:
    with st.chat_message("assistant"):
        st.write("Welcome to your AI Yoga Guide! Let's begin.")
        
        st.write("### 🍎 Food & Water Precautions")
        st.write("- Avoid eating heavy meals 2-3 hours before yoga.")
        st.write("- Stay hydrated, but avoid drinking too much water just before the session.")
        st.write("- Light snacks (fruits, nuts) are okay 30 minutes before yoga.")
        
        if st.button("Start Chat"):
            st.session_state.chat_step = 1
            st.rerun()


# Step 1: Ask for yoga level
if st.session_state.chat_step == 1:
    with st.chat_message("assistant"):
        st.write("What is your level in yoga?")
        yoga_level = st.selectbox("Choose your level:", ["Beginner", "Intermediate", "Advanced", "No Idea"])
        if st.button("Submit"):
            st.session_state.user_data["yoga_level"] = yoga_level
            st.session_state.chat_step = 2
            st.rerun()

# Step 2: Show beginner videos if applicable
if st.session_state.chat_step == 2:
    with st.chat_message("assistant"):
        if st.session_state.user_data["yoga_level"] in ["Beginner", "No Idea"]:
            st.write("Since you're a beginner, here are some easy poses to start with:")
            st.video("https://www.youtube.com/watch?v=NYhH8Gr35cI")  # Mountain Pose
            st.video("https://www.youtube.com/watch?v=149Iac5fmoE")  # basic yoga part 1
            st.video("https://www.youtube.com/watch?v=aCuhhN5HTFg")  # basic yoga part 2
        else:
            st.write("You're experienced! Let's move on.")
        if st.button("Next"):
            st.session_state.chat_step = 3
            st.rerun()

# Step 3: Collect User Information
if st.session_state.chat_step == 3:
    with st.chat_message("assistant"):
        st.write("Before we start, let's check some health precautions.")

    if "age" not in st.session_state.user_data:
        age = st.text_input("Enter your age:", key="age_input")
        if age:
            st.session_state.user_data["age"] = age
            st.rerun()
    elif "injuries" not in st.session_state.user_data:
        injuries = st.multiselect("Select any injuries you have:",
             ["Knee", "Wrist"],key="injury_input")
        if st.button("Next"):
            st.session_state.user_data["injuries"] = injuries
            st.rerun()
    elif "medical_conditions" not in st.session_state.user_data:
        conditions = st.multiselect(
            "Do you have any medical conditions?",
            ["None", "Pregnancy", "Sciatica", "Herniated Disc", "Hypertension", "Arthritis"],
            key="medical_condition_input"
        )
        if st.button("Next"):
            st.session_state.user_data["medical_conditions"] = conditions
            st.rerun()

    elif "sub_conditions" not in st.session_state.user_data:
        sub_conditions = {}
        if "Hypertension" in st.session_state.user_data["medical_conditions"]:
            sub_conditions["Hypertension"] = st.selectbox(
                "Which type of Hypertension do you have?",
                ["Essential Hypertension", "Secondary Hypertension"],
                key="hypertension_sub"
            )
        if "Sciatica" in st.session_state.user_data["medical_conditions"]:
            sub_conditions["Sciatica"] = st.selectbox(
                "What kind of Sciatica do you have?",
                ["Acute Sciatica", "Chronic Sciatica"],
                key="sciatica_sub"
            )
        if "Herniated Disc" in st.session_state.user_data["medical_conditions"]:
            sub_conditions["Herniated Disc"] = st.selectbox(
                "Which region is affected?",
                ["Cervical Herniation (Neck)", "Lumbar Herniation (Lower Back)"],
                key="herniated_sub"
            )
        if st.button("Submit Health Info"):
            st.session_state.user_data["sub_conditions"] = sub_conditions
            st.session_state.chat_step = 4
            st.rerun()

            
# Step 4: ML-based Pose Classification + Explainability
if st.session_state.chat_step == 4:
    st.header("🧠 AI-Powered Pose Safety & Explanation")

    poses_to_check = [
        "Child’s Pose", "Bound Angle Pose", "Cat-Cow Pose",
        "Downward Facing Dog Pose", "Mountain Pose", "Angle Pose"
    ]

    user_data = st.session_state.get("user_data", {})
    conditions = user_data.get("medical_conditions", [])
    injuries = user_data.get("injuries", [])
    yoga_level = user_data.get("yoga_level", "Beginner")

    safe_poses = []
    unsafe_poses = []

    # Iterate over each pose and categorize it as safe or unsafe
    for pose in poses_to_check:
        prediction, explanation = generate_pose_prediction_and_explanation(pose, conditions, injuries, yoga_level)

        # Use string comparison for SAFE/UNSAFE
        if prediction == "SAFE":
            safe_poses.append(pose)
        else:
            unsafe_poses.append(pose)

        # Display explanation for each pose
        st.markdown(explanation)
        st.markdown("---")

    # Check and display the results
    if safe_poses:
        st.success(f"✅ Safe Poses: {', '.join(safe_poses)}")
    else:
        st.success("No poses marked safe.")
        
    if unsafe_poses:
        st.error(f"🚫 Unsafe Poses: {', '.join(unsafe_poses)}")
    else:
        st.error("None unsafe.")

    # Pose selection if safe poses exist
    if safe_poses:
        st.selectbox("Select a pose to perform:", safe_poses, key="selected_pose")

    # Move to next step
    if st.button("Proceed to Pre-Asana Precautions"):
        st.session_state.chat_step = 5
        st.rerun()


# Step 5: Display Pre-Asana Precautions
if st.session_state.chat_step == 5 and st.session_state.selected_pose:
    pose = st.session_state.selected_pose

    precautions = {
        "Child’s Pose": [
            "Avoid if you have knee injuries or severe back pain.",
            "People with high blood pressure should avoid keeping their head too low.",
        ],
        "Bound Angle Pose": [
            "If you have a hip injury, avoid pressing the knees too hard.",
            "People with lower back pain should sit with support under the hips.",
        ],
        "Cat-Cow Pose": [
            "Avoid if you have a neck injury or pain.",
            "People with herniated discs should avoid deep backbends.",
        ],
    }

    st.write(f"### ✅ Pre-Asana Precautions for {pose}:")
    for point in precautions.get(pose, ["No specific precautions available."]):
        st.write(f"- {point}")

    if st.button("Lets start the streches and yoga pose"):
        st.session_state.chat_step = 6
        st.rerun()

# Step 6: Warm-up Stretches (Common for all levels)
if st.session_state.chat_step == 6:
    st.write("### 🏃 Warm-Up Stretches")
    st.video("https://www.youtube.com/watch?v=2q3HwR-HILc")  # General Warm-up Stretches
    st.video("https://www.youtube.com/watch?v=DCQwQYRwBWE")  # Neck & Shoulder Warm-up
    st.video("https://www.youtube.com/watch?v=IMzxosIffag")  # Spinal & Full Body Warm-up

    if st.button("Proceed to Yoga Pose"):
        st.session_state.chat_step = 7
        st.rerun()

# Step 7: Show final yoga session videos
if st.session_state.chat_step == 7:
    with st.chat_message("assistant"):
        st.write("You're all set! Follow along with your personalized yoga session:")

        pose_videos = {
            "Child’s Pose": "https://www.youtube.com/watch?v=eqVMAPM00DM",
            "Bound Angle Pose": "https://www.youtube.com/watch?v=B6tb4TncKhY",
            "Cat-Cow Pose": "https://www.youtube.com/watch?v=y39PrKY_4JM",
            "Downward Facing Dog": "https://www.youtube.com/watch?v=j97SSGsnCAQ",
            "Mountain Pose": "https://www.youtube.com/watch?v=NYhH8Gr35cI",
            "Angle Pose": "https://www.youtube.com/watch?v=0lfzG9jH6cM"
        }

        safe_poses = st.session_state.get("safe_poses", [])

        for pose, url in pose_videos.items():
            if pose in safe_poses:
                st.write(f"### {pose}")
                st.video(url)

    if st.button("Want to ask something else about Yoga?"):
        st.session_state.chat_step = 8
        st.rerun()


# Step 8: Yoga Q&A using Groq + Mixtral
if st.session_state.chat_step == 8:
    from utils.groq_rag import answer_with_groq

    # Load yoga knowledge only once
    if "yoga_knowledge" not in st.session_state:
        with open("yoga_knowledge.txt", "r", encoding="utf-8") as f:
            st.session_state.yoga_knowledge = f.read()

    st.title("💬 Ask Any Yoga Question")

    st.subheader("🧘 Type your question:")
    custom_query = st.text_input("Ask me anything related to yoga, poses, safety, breathing, benefits...")

    if custom_query:
        with st.spinner("Mixtral (Groq) is thinking..."):
            response = answer_with_groq(custom_query, st.session_state.yoga_knowledge)
        st.markdown("### 🤖 Mixtral (Groq) Response")
        st.write(response)

    st.markdown("---")
    st.subheader("📚 Or Pick from Common Questions")

    faq_questions = [
        "Is Child’s Pose safe during pregnancy?",
        "Can people with knee injuries do Bound Angle Pose?",
        "Is Cobra Pose good for sciatica?",
        "What are the benefits of Cat-Cow Pose?",
        "Should people with herniated discs avoid Downward Facing Dog?",
        "What precautions should be taken for Warrior I Pose?",
        "Can I do yoga if I have arthritis?",
        "Which poses are best for hypertension?",
        "What are the safest poses for beginners?",
        "How does Tree Pose improve balance?",
    ]

    selected_faq = st.selectbox("Choose a FAQ:", faq_questions)

    if st.button("Ask This Question"):
        with st.spinner("Mixtral is generating your answer..."):
            response = answer_with_groq(selected_faq, st.session_state.yoga_knowledge)
        st.markdown("### 🤖 Mixtral (Groq) Response")
        st.write(response)















