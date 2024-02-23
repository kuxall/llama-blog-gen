import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


def getLLAMAresponse(in_text, threshold, blog_style):
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama')

    # Let's set the prompt template
    template = """
		Write a {blog_style} blog post about {in_text} within {threshold} words.

		Title: [Your Title Here]

		Introduction:
		[Provide a brief introduction to the topic and why it's important or relevant.]

		Body:
		[Discuss key points, insights, examples, and tips related to {in_text}. You can break down the content into subsections for better readability.]

		Conclusion:
		[Summarize the main points discussed in the blog post and provide any concluding remarks or calls to action.]

		Word Count: {threshold}	
	"""
    prompt = PromptTemplate(
        input_variables=['blog_style', 'in_text', 'threshold'], template=template)
    response = llm.invoke(prompt.format(blog_style=blog_style,
                                        in_text=in_text, threshold=threshold))
    print(response)
    return response


st.set_page_config(
    page_title="BlogGen",
    page_icon="📝✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header('BLOG GENERATOR 📝✨')

# Now, Let's select the niche for blog
in_text = st.text_input("Enter the blog topic")

column1, column2 = st.columns([8, 8])

with column1:
    threshold = st.text_input("Enter the threshold for generating blogs")

with column2:
    blog_style = st.selectbox(' We are writing for ...',
                              ('Students', 'Teachers', 'Data Scientists', 'Researchers'), index=0)

submit = st.button("Generate 📝")

if submit:
    st.write(getLLAMAresponse(in_text, threshold, blog_style))
