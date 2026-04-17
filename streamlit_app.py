import streamlit as st
from main import RecipeRAGSystem



@st.cache_resource
def init_system():
    #构建系统
    system = RecipeRAGSystem()
    #初始化系统
    system.initialize_system()
    system.build_knowledge_base()
    return system

def streamlit():
    st.markdown("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")#制定聊天框标题
    # 【注入 CSS 黑科技：强行修改标题大小】
    st.markdown("""
    <style>
    /* 专门针对聊天记录里的各级标题进行缩放 */
    .stChatMessage h1 {
        font-size: 1.5rem !important; /* 一级标题：原来大概是 2.5rem */
        padding-bottom: 0.3rem;
    }
    .stChatMessage h2 {
        font-size: 1.3rem !important; /* 二级标题：原来大概是 2.0rem */
        padding-bottom: 0.3rem;
    }
    .stChatMessage h3 {
        font-size: 1.1rem !important; /* 三级标题：原来大概是 1.75rem */
        padding-bottom: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    #判断是否存在聊天库
    if "message" not in st.session_state:
        st.session_state.message = []#没有就初始化聊天库
    if 'system' not in st.session_state:
        st.session_state.system = init_system()
    # 搭建聊天窗口
    # 建立容器 高度为500px
    messages = st.container(height=600)
    # 把历史记录加载到聊天框中
    for message in st.session_state.message:
        with messages.chat_message(message[0]):
            st.write(message[1])
    if use_input := st.chat_input("说说想吃点什么吧..."):#这里使用:=将后面的值直接赋给prompt同时判断真假
        st.session_state.message.append(('human', use_input))#直接将用户的输入传入聊天记录库中
        #显示用户输入
        with messages.chat_message('human'):
            st.write(use_input)
        answer = st.session_state.system.ask_question(question = use_input, stream = True)
        st.session_state.message.append(('ai', answer))
        with messages.chat_message("ai"):
            st.write(answer)

if __name__ == '__main__':
    streamlit()