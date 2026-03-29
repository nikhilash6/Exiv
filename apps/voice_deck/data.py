SPEAKERS = {
    "Vivian": {
        "language": "Chinese",
        "description": "Bright, slightly edgy young female voice",
        "text": "你好，我是Vivian。我是一个性格开朗、略带锋芒的年轻女性。很高兴认识你！",
    },
    "Serena": {
        "language": "Chinese", 
        "description": "Warm, gentle young female voice",
        "text": "你好，我是Serena。我的声音温暖而温柔，希望能为你带来舒适的体验。",
    },
    "Uncle_Fu": {
        "language": "Chinese",
        "description": "Seasoned male voice with a low, mellow timbre",
        "text": "你好，我是福叔。我是一个声音低沉、醇厚的成熟男性，有什么可以帮你的吗？",
    },
    "Dylan": {
        "language": "Chinese",
        "description": "Youthful Beijing male voice with a clear, natural timbre (Beijing Dialect)",
        "text": "嘿，我是Dylan。我是一个北京小伙子，说话带着京味儿，声音清晰自然。",
    },
    "Eric": {
        "language": "Chinese",
        "description": "Lively Chengdu male voice with a slightly husky brightness (Sichuan Dialect)",
        "text": "你好噻，我是Eric。我是个成都小伙儿，声音有点沙哑但很亮堂，巴适得板！",
    },
    "Ryan": {
        "language": "English",
        "description": "Dynamic male voice with strong rhythmic drive",
        "text": "Hello, I'm Ryan. I have a dynamic voice with strong rhythmic drive. How can I help you today?",
    },
    "Aiden": {
        "language": "English",
        "description": "Sunny American male voice with a clear midrange",
        "text": "Hello there! My name is Aiden. I'm here to help you with any questions you might have. I can speak in a natural, conversational tone for extended periods of time.",
    },
    "Ono_Anna": {
        "language": "Japanese",
        "description": "Playful Japanese female voice with a light, nimble timbre",
        "text": "こんにちは、小野アンナです。明るく軽やかな声が特徴の日本女性です。よろしくお願いします！",
    },
    "Sohee": {
        "language": "Korean",
        "description": "Warm Korean female voice with rich emotion",
        "text": "안녕하세요, 소희입니다. 따뜻하고 감성적인 한국어 여성 목소리입니다. 만나서 반갑습니다!",
    },
}

# Cross-lingual examples: each speaker speaking English
CROSS_LINGUAL_TEXT = {
    "Vivian": "Hello, this is Vivian speaking in English. I can speak multiple languages fluently.",
    "Serena": "Hello, this is Serena. Though my native language is Chinese, I can also speak English naturally.",
    "Uncle_Fu": "Hello, I'm Uncle Fu. Even as a Chinese native speaker, I can communicate with you in English.",
    "Dylan": "Hey there! Dylan here. I may be from Beijing, but I can totally speak English too!",
    "Eric": "Hi! Eric speaking. This is me speaking English with my unique voice from Chengdu.",
    "Ryan": "Hi, Ryan here. I'm a native English speaker with a dynamic, rhythmic voice.",
    "Aiden": "Hello! Aiden speaking. As a native English speaker, I'm here to assist you with anything you need.",
    "Ono_Anna": "Hello! This is Ono Anna speaking English. Even though I'm Japanese, I can speak English well!",
    "Sohee": "Hello, Sohee here. I'm a Korean speaker, but I can also communicate with you in English.",
}

# Dialect demonstrations - showcasing regional Chinese dialects
DIALECT_DEMOS = {
    "Standard_Mandarin": {
        "speaker": "Vivian",
        "language": "Chinese",
        "text": "大家好，这是标准的普通话。我们可以用清晰的标准中文进行交流。",
        "description": "Standard Mandarin Chinese"
    },
    "Beijing_Dialect": {
        "speaker": "Dylan", 
        "language": "Chinese",
        "text": "咱北京人啊，说话就这个味儿！您听听，这京腔京韵的多地道啊。",
        "description": "Beijing Dialect (Northern Chinese)"
    },
    "Sichuan_Dialect": {
        "speaker": "Eric",
        "language": "Chinese", 
        "text": "要得嘛，我们四川人说话就是这么巴适！你听我这川普，安逸得很噻！",
        "description": "Sichuan Dialect (Southwestern Chinese)"
    },
}

# Multi-language support demos - showing all supported languages
MULTI_LANGUAGE_DEMOS = [
    {"language": "Chinese", "text": "这是中文普通话测试。你好，世界！"},
    {"language": "English", "text": "This is English language testing. Hello, world!"},
    {"language": "Japanese", "text": "これは日本語のテストです。こんにちは、世界！"},
    {"language": "Korean", "text": "이것은 한국어 테스트입니다. 안녕하세요, 세계!"},
    {"language": "German", "text": "Dies ist ein deutscher Test. Hallo Welt!"},
    {"language": "French", "text": "Ceci est un test en français. Bonjour le monde!"},
    {"language": "Spanish", "text": "Esta es una prueba en español. ¡Hola mundo!"},
    {"language": "Italian", "text": "Questo è un test in italiano. Ciao mondo!"},
    {"language": "Portuguese", "text": "Este é um teste em português. Olá mundo!"},
    {"language": "Russian", "text": "Это тест на русском языке. Привет, мир!"},
]