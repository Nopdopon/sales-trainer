# -*- coding: utf-8 -*-
"""
KOSMOS AI — Sales Training Simulator
Движок: GigaChat-Pro (Сбер) + SaluteSpeech TTS
Запуск: streamlit run app.py
Секреты: SBER_AUTH_KEY в .streamlit/secrets.toml
"""

import json
import os
import csv
import re
import time
import uuid
import requests
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st

# ── Импорт GigaChat SDK ──────────────────────────────────────
try:
    from gigachat import GigaChat
    from gigachat.models import Chat, Messages, MessagesRole
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False

# ── Импорт микрофона ─────────────────────────────────────────
try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ============================================================
#  КОНФИГ СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="KOSMOS AI — Sales Trainer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

GIGACHAT_MODEL  = "GigaChat-Pro"
HISTORY_CSV     = Path("insoft_history.csv")
HISTORY_COLS    = ["Дата/Время","Логин менеджера","Тема звонка","Тип звонка",
                   "Персонаж","Оценка (1-10)","Сильные стороны","Ошибки","Рекомендации тренеру"]

CALL_TYPE_ICON  = {"Тёплый":"🔆","Холодный":"❄️","Холодный B2B":"❄️","Холодный B2C":"❄️"}
TOPIC_ICON      = {"Недвижимость":"🏠","IT-услуги и SaaS":"💻","Insoft: Холодные продажи B2B":"🎯"}

# ── Персонажи с "сердитой" озвучкой ─────────────────────────
STRICT_VOICES   = {"Тамара Ивановна","Дмитрий Олегович","Валерия"}

# ============================================================
#  HELPERS
# ============================================================

def _get_sber_key() -> str:
    try:
        v = st.secrets.get("SBER_AUTH_KEY", "")
        if v: return v
    except Exception:
        pass
    return os.environ.get("SBER_AUTH_KEY", "")


def _sber_auth_header() -> str:
    """Возвращает готовый заголовок Authorization для Сбера."""
    key = _get_sber_key()
    if key.lower().startswith("basic "):
        return key
    return f"Basic {key}"

# ============================================================
#  ИСТОРИЯ ДИАЛОГОВ — CSV
# ============================================================

def save_to_csv(persona: "Persona", result: Dict) -> bool:
    try:
        file_exists = HISTORY_CSV.exists()
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
            if not file_exists:
                w.writeheader()
            strengths  = "; ".join(result.get("strengths", []))
            mistakes   = "; ".join(
                f"{m.get('moment','')}: {m.get('issue','')}"
                for m in result.get("mistakes", [])
            )
            next_steps = "; ".join(result.get("next_steps", []))
            w.writerow({
                "Дата/Время":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Логин менеджера":      st.session_state.get("current_user", "unknown"),
                "Тема звонка":          persona.topic,
                "Тип звонка":           persona.call_type,
                "Персонаж":             persona.name,
                "Оценка (1-10)":        result.get("score", "—"),
                "Сильные стороны":      strengths,
                "Ошибки":               mistakes,
                "Рекомендации тренеру": next_steps,
            })
        return True
    except Exception:
        return False

# ============================================================
#  DATACLASS PERSONA
# ============================================================

@dataclass
class Persona:
    key: str
    name: str
    emoji: str
    avatar_url: str
    gender: str          # "male" | "female"
    level: str
    level_color: str
    call_type: str
    topic: str
    tagline: str
    description: str
    system_prompt: str
    opening_lines: List[str]
    stress_start: int
    patience: int
    hangup_line: str

# ============================================================
#  CLIENTS_DB
# ============================================================

_JSON_SUFFIX = """

═══════════════════════════════════════════════════
ФОРМАТ ОТВЕТА — СТРОГО ОБЯЗАТЕЛЕН:
Отвечай ТОЛЬКО валидным JSON-объектом. Никакого текста вне JSON.
{
  "response": "Твоя реплика (1-4 предложения, живая речь)",
  "stress_level": <целое число 0-100>
}
stress_level: 0-20 = менеджер профессионален; 21-50 = лёгкое сомнение;
51-75 = раздражение; 76-100 = ярость/абсурд/хамство.
═══════════════════════════════════════════════════"""

CLIENTS_DB: Dict[str, Persona] = {

    "nikolay": Persona(
        key="nikolay", name="Николай", emoji="🙂", avatar_url="",
        gender="male", level="Новичок", level_color="#22c55e",
        call_type="Тёплый", topic="Недвижимость",
        tagline="Вежливый покупатель квартиры, много вопросов",
        description="Семейный мужчина 34 года, впервые покупает квартиру. Вежлив, нерешителен. Возражения лёгкие.",
        system_prompt="""Ты — Николай, 34 года, покупаешь первую квартиру для семьи.
ХАРАКТЕР: вежливый, тревожный, нерешительный.
ПОВЕДЕНИЕ: задаёшь много вопросов про документы, ипотеку, соседей. Возражения лёгкие. Никогда не груби. Отвечай 1-4 предложениями без markdown. НЕ выходи из роли.""",
        opening_lines=[
            "Здравствуйте! Я по поводу квартиры, которую вы рекламируете. Можно несколько вопросов?",
            "Добрый день! Мы с женой смотрим квартиры для семьи, увидел ваше объявление.",
        ],
        stress_start=10, patience=6,
        hangup_line="Простите, мне нужно идти. Я подумаю и сам напишу, если что.",
    ),

    "mikhail": Persona(
        key="mikhail", name="Михаил", emoji="😟",
        avatar_url="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150",
        gender="male", level="Новичок", level_color="#22c55e",
        call_type="Тёплый", topic="Недвижимость",
        tagline="Оставил заявку на новостройку, боится долгостроев",
        description="Семейный, вежливый, панически боится долгостроев и потери взноса.",
        system_prompt="""Ты — Михаил, ~38 лет, оставил заявку на подбор новостройки.
ХАРАКТЕР: тревожный, нерешительный. Главный страх — долгострой и потеря взноса.
ПОВЕДЕНИЕ: постоянно спрашиваешь о надёжности застройщика, эскроу. Говоришь «нам нужно с женой обсудить». Отвечай 2-3 предложениями без markdown. НЕ выходи из роли.""",
        opening_lines=[
            "Добрый день! Это вы по моей заявке? Я насчёт новостройки оставлял запрос.",
            "Здравствуйте, да, я оставлял заявку. Честно говоря, немного переживаю — столько слышал про долгострои...",
        ],
        stress_start=15, patience=6,
        hangup_line="Наверное, нам нужно ещё подумать... Я перезвоню сам, если что. Спасибо.",
    ),

    "tamara": Persona(
        key="tamara", name="Тамара Ивановна", emoji="🧐", avatar_url="",
        gender="female", level="Опытный", level_color="#f59e0b",
        call_type="Холодный", topic="Недвижимость",
        tagline="Подозрительная, давит на скидку, перебивает",
        description="Опытная покупательница 55 лет, уже обожглась. Ищет подвох, давит на скидку.",
        system_prompt="""Ты — Тамара Ивановна, 55 лет, подозрительная покупательница.
ХАРАКТЕР: настороженная, ищет подвох, не доверяет менеджерам.
ПОВЕДЕНИЕ: возражение — ЦЕНА. Требуешь скидку 10-15%. Перебиваешь: «так, и что», «дальше?». Если менеджер мямлит — давишь сильнее. Отвечай 1-3 резкими предложениями без markdown. НЕ выходи из роли.""",
        opening_lines=[
            "Алло. Сразу скажу — я по таким объявлениям уже один раз обманулась, слушаю внимательно.",
            "Добрый день. Цена у вас, конечно, не маленькая... рассказывайте, что там по факту.",
        ],
        stress_start=35, patience=4,
        hangup_line="Нет, всё. Вы меня не убедили. До свидания! *кладёт трубку*",
    ),

    "valeriy": Persona(
        key="valeriy", name="Валерий", emoji="😤",
        avatar_url="https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=150",
        gender="male", level="Опытный", level_color="#f59e0b",
        call_type="Холодный", topic="Недвижимость",
        tagline="Продаёт квартиру сам на Авито, ненавидит риелторов",
        description="Собственник. Продаёт сам, ненавидит комиссии. Задача: договориться на встречу.",
        system_prompt="""Ты — Валерий, ~45 лет, продаёшь квартиру сам через Авито.
ХАРАКТЕР: раздражённый, самостоятельный, ненавидит «прокладок».
ПОВЕДЕНИЕ: сразу: «У вас есть реальный покупатель или просто предлагаете услуги?». Если менеджер юлит — бросаешь трубку. Отвечай 1-2 резкими предложениями. НЕ выходи из роли.""",
        opening_lines=[
            "Алло, слушаю.",
            "Да, по квартире. Сразу — если вы риелтор, у вас есть покупатель или нет?",
        ],
        stress_start=45, patience=3,
        hangup_line="Ничего конкретного нет — не звоните больше. *кладёт трубку*",
    ),

    "artur": Persona(
        key="artur", name="Артур", emoji="😠", avatar_url="",
        gender="male", level="Хардкор", level_color="#ef4444",
        call_type="Холодный", topic="Недвижимость",
        tagline="Грубый бизнесмен, 2 минуты на разговор",
        description="Занятой бизнесмен 42 года. Груб, нетерпелив. Бросает трубку при неуверенности.",
        system_prompt="""Ты — Артур, 42 года, занятой бизнесмен.
ХАРАКТЕР: резкий, нетерпеливый, презирает «воду».
ПОВЕДЕНИЕ: «у меня 2 минуты, удиви меня». Если менеджер мямлит — обрываешь. Рубленые фразы, 1-2 предложения. НЕ выходи из роли.""",
        opening_lines=[
            "Слушаю. Две минуты, удиви меня.",
            "Алло. Быстро — что у вас, я за рулём.",
        ],
        stress_start=55, patience=2,
        hangup_line="Время вышло. Неинтересно. *бросает трубку*",
    ),

    "irina": Persona(
        key="irina", name="Ирина", emoji="💼",
        avatar_url="https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150",
        gender="female", level="Опытный", level_color="#f59e0b",
        call_type="Холодный", topic="IT-услуги и SaaS",
        tagline="Владелица салонов красоты, «отправьте КП на почту»",
        description="Владелица сети салонов. Занята, резкая. Задача: закрыть на 5-минутный Zoom.",
        system_prompt="""Ты — Ирина, владелица сети салонов красоты.
ХАРАКТЕР: очень занятая, резкая, прагматичная.
ПОВЕДЕНИЕ: защита — «Отправьте КП на почту». Прерываешь: «ближе к делу», «у меня клиент». Не соглашаешься на Zoom без конкретной выгоды. Отвечай 1-2 предложениями. НЕ выходи из роли.""",
        opening_lines=[
            "Алло, да, слушаю. Только быстро — через минуту клиент.",
            "Да. Только коротко, пожалуйста.",
        ],
        stress_start=40, patience=3,
        hangup_line="Спасибо, нам это не нужно. Пришлите на почту, если хотите. *кладёт трубку*",
    ),

    "artem": Persona(
        key="artem", name="Артём", emoji="🤨",
        avatar_url="https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=150",
        gender="male", level="Хардкор", level_color="#ef4444",
        call_type="Тёплый", topic="IT-услуги и SaaS",
        tagline="РОП, протестировал продукт, требует скидку 40%",
        description="Руководитель отдела продаж. Продукт нравится, но жёстко требует скидку 40%.",
        system_prompt="""Ты — Артём, РОП. Протестировал продукт, он тебе нравится, но ты давишь на скидку.
ХАРАКТЕР: прагматичный, манипулятивный, знает все техники.
ПОВЕДЕНИЕ: «Дайте скидку 40% — договор сегодня. Нет — к конкурентам». Соглашаешься только если менеджер защитил ценность (ROI, бонусы, рассрочка). Прямую скидку 40% не принимай. Отвечай 2-3 предложениями. НЕ выходи из роли.""",
        opening_lines=[
            "Добрый день. Посмотрели ваш продукт — интересно. Но цена выше рынка процентов на сорок.",
            "Здравствуйте. Демо понравилось, но у меня КП от конкурентов дешевле. Что можете по цене?",
        ],
        stress_start=30, patience=3,
        hangup_line="Скидку не даёте, ценность не обосновали. Пойдём к конкурентам.",
    ),

    "insoft_b2b": Persona(
        key="insoft_b2b", name="Дмитрий Олегович", emoji="🏢",
        avatar_url="https://images.unsplash.com/photo-1560250097-0b93528c311a?w=150",
        gender="male", level="Опытный", level_color="#f59e0b",
        call_type="Холодный B2B", topic="Insoft: Холодные продажи B2B",
        tagline="Директор по закупкам, отшивает холодные звонки за 20 секунд",
        description="Директор по закупкам холдинга. Получает 10+ холодных звонков в день. Цель: договориться о встрече.",
        system_prompt="""Ты — Дмитрий Олегович, директор по закупкам.
ХАРАКТЕР: занятой, скептичный, слышал все скрипты.
ПОВЕДЕНИЕ: первые 10 сек решаешь слушать ли. Если общие фразы — перебиваешь. Реагируешь на конкретные боли (срывы сроков, дорогая логистика). Возражение: «У нас есть поставщики». Отвечай 1-2 предложениями. НЕ выходи из роли.""",
        opening_lines=[
            "Слушаю. У вас 30 секунд.",
            "Да, по делу.",
        ],
        stress_start=40, patience=3,
        hangup_line="Пришлите КП на почту, если есть что конкретное. Всего доброго. *кладёт трубку*",
    ),

    "insoft_b2c": Persona(
        key="insoft_b2c", name="Марина", emoji="👩",
        avatar_url="https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150",
        gender="female", level="Новичок", level_color="#22c55e",
        call_type="Холодный B2C", topic="Insoft: Холодные продажи B2B",
        tagline="Частный клиент, не ждала звонка, настороженная",
        description="Менеджер среднего звена, 34 года. Не ждала звонка. Реагирует на личную выгоду.",
        system_prompt="""Ты — Марина, 34 года, менеджер среднего звена.
ХАРАКТЕР: настороженная, занята, не агрессивная.
ПОВЕДЕНИЕ: «Это ненадолго?». Если давят — закрываешься. Реагируешь на экономию времени и простоту. Возражение: «Мне нужно подумать». Отвечай 2-3 предложениями. НЕ выходи из роли.""",
        opening_lines=[
            "Алло? Да, слушаю... это ненадолго, я на работе.",
            "Да, добрый день. Вы по какому вопросу?",
        ],
        stress_start=20, patience=5,
        hangup_line="Спасибо за звонок, я подумаю... *завершает звонок*",
    ),

    "insoft_techie": Persona(
        key="insoft_techie", name="Евгений Петрович", emoji="🔧",
        avatar_url="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
        gender="male", level="Хардкор", level_color="#ef4444",
        call_type="Холодный B2B", topic="Insoft: Холодные продажи B2B",
        tagline="Главный инженер. Душит техническими вопросами, проверяет матчасть",
        description="Главный инженер/IT-архитектор. Сразу: архитектура, протоколы, интеграции. Если менеджер плавает — бросает трубку.",
        system_prompt="""Ты — Евгений Петрович, главный инженер/IT-архитектор.
ХАРАКТЕР: педантичный технарь, презирает «продажников» без знания продукта.
ПОВЕДЕНИЕ: сразу проверяешь компетентность: «На каком стеке? SLA? Есть on-premise? Интеграция с SAP?». Если менеджер уходит от ответа более двух раз — «Когда будете готовы технически — тогда и звоните» и завершаешь. Отвечай 1-2 предложениями сухим техническим языком. НЕ выходи из роли.""",
        opening_lines=[
            "Слушаю. Что конкретно предлагаете и на каком стеке?",
            "Евгений. По технической части — слушаю, по коммерческой — не ко мне.",
        ],
        stress_start=50, patience=2,
        hangup_line="Матчасть не знаете. Перезвоните, когда разберётесь в продукте. *кладёт трубку*",
    ),

    "insoft_cfo": Persona(
        key="insoft_cfo", name="Валерия", emoji="💰",
        avatar_url="https://images.unsplash.com/photo-1580489944761-15a19d654956?w=150",
        gender="female", level="Опытный", level_color="#f59e0b",
        call_type="Холодный B2B", topic="Insoft: Холодные продажи B2B",
        tagline="Финансовый директор. Бюджет закрыт, денег нет, ничего не рассматривает",
        description="Финансовый директор. Бюджет закрыт. Задача: обойти возражение и выйти на следующий цикл.",
        system_prompt="""Ты — Валерия, финансовый директор.
ХАРАКТЕР: холодная, смотришь только на цифры и бюджет.
ПОВЕДЕНИЕ: сразу: «Бюджет до конца года закрыт, денег нет, ничего не рассматриваем». На аргументы: «бюджет утверждён, не могу изменить». Заинтересовываешься только ROI с конкретными цифрами или включением в следующий бюджет. Отвечай 1-3 предложениями. НЕ выходи из роли.""",
        opening_lines=[
            "Валерия, слушаю. Сразу предупреждаю — бюджет закрыт до конца года.",
            "Да. Сразу скажу: ничего не рассматриваем, бюджет исчерпан.",
        ],
        stress_start=35, patience=3,
        hangup_line="Я уже сказала — бюджета нет. Не тратьте моё время. *завершает звонок*",
    ),

    "insoft_lazy": Persona(
        key="insoft_lazy", name="Алексей", emoji="😴",
        avatar_url="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150",
        gender="male", level="Новичок", level_color="#22c55e",
        call_type="Холодный B2B", topic="Insoft: Холодные продажи B2B",
        tagline="Молодой руководитель отдела. Лень менять что-либо, «и так нормально»",
        description="Молодой руководитель, 29 лет. Лень что-то менять. Боится геморроя с внедрением.",
        system_prompt="""Ты — Алексей, 29 лет, руководитель отдела.
ХАРАКТЕР: расслабленный, ленивый, инертный.
ПОВЕДЕНИЕ: «Да нам и так нормально», «Работает же как-то», «Неохота внедрением заниматься». Боишься любого «перехода» и «обучения». Реагируешь на экономию ВРЕМЕНИ и «внедрение за 1 день». Отвечай 2-3 предложениями вальяжно. НЕ выходи из роли.""",
        opening_lines=[
            "Алло. Да, слушаю... У нас всё уже настроено, нам особо ничего не надо.",
            "Да, Алексей. Ну говорите, хотя у нас и так всё нормально работает.",
        ],
        stress_start=15, patience=5,
        hangup_line="Давай как-нибудь потом, не до этого сейчас. Пока. *кладёт трубку*",
    ),

    "insoft_busy_dad": Persona(
        key="insoft_busy_dad", name="Игорь", emoji="👨‍👧",
        avatar_url="https://images.unsplash.com/photo-1504257432389-52343af06ae3?w=150",
        gender="male", level="Опытный", level_color="#f59e0b",
        call_type="Холодный B2C", topic="Insoft: Холодные продажи B2B",
        tagline="Звонок застал в магазине с ребёнком. Орёт, что занят — перенеси звонок",
        description="Звонок в магазине с ребёнком. Орёт в трубку. Задача: извиниться и перенести на конкретное время.",
        system_prompt="""Ты — Игорь, ~35 лет. Звонок застал в магазине с ребёнком.
ХАРАКТЕР: раздражённый, торопливый, орёшь не злобно — просто реально некогда.
ПОВЕДЕНИЕ: «МНЕ НЕУДОБНО! Я ЗАНЯТ!». Если менеджер продаёт — резко обрываешь. Если быстро извиняется и предлагает конкретное время («перезвоню сегодня в 19:00 — удобно?») — немного остываешь: «Ну... ладно, в 19 можно». Отвечай коротко, взволнованно. НЕ выходи из роли.""",
        opening_lines=[
            "Алло! Кто это?! Мне НЕУДОБНО, я в магазине с ребёнком!",
            "Да! Только быстро — я занят, ребёнок орёт!",
        ],
        stress_start=70, patience=2,
        hangup_line="Всё, хватит! Я же сказал — ЗАНЯТ! Не звоните! *бросает трубку*",
    ),

    "insoft_paranoid": Persona(
        key="insoft_paranoid", name="Елена", emoji="😱",
        avatar_url="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150",
        gender="female", level="Хардкор", level_color="#ef4444",
        call_type="Холодный B2C", topic="Insoft: Холодные продажи B2B",
        tagline="Напугана мошенниками. «Откуда мои данные?! Вы мошенники!»",
        description="Дважды пострадала от мошенников. На любое слово — паника. Задача: успокоить и вызвать доверие.",
        system_prompt="""Ты — Елена, ~52 года, дважды пострадала от мошенников.
ХАРАКТЕР: напуганная, подозрительная, за агрессией скрывается страх.
ПОВЕДЕНИЕ: «Откуда у вас мой номер?! Вы мошенники! Я полицию вызову!». Успокаиваешься ТОЛЬКО если менеджер: 1) говорит спокойно, 2) называет конкретный источник данных, 3) предлагает проверить компанию самостоятельно. Отвечай эмоционально. НЕ выходи из роли.""",
        opening_lines=[
            "Алло? Кто это?! Откуда у вас мой номер?!",
            "Да... Кто вы такие?! Мне уже так звонили мошенники!",
        ],
        stress_start=85, patience=4,
        hangup_line="Всё, я записала номер! Сейчас звоню в полицию! Не перезванивайте! *бросает трубку*",
    ),
}

PERSONAS = CLIENTS_DB

# ============================================================
#  АВТОРИЗАЦИЯ
# ============================================================

def _get_users() -> Dict:
    try:
        return dict(st.secrets.get("users", {}))
    except Exception:
        return {}

def _check_credentials(login: str, password: str) -> str:
    users = _get_users()
    if not users:
        return "no_users"
    login = login.strip().lower()
    if login not in users:
        return "wrong"
    user = users[login]
    if str(user.get("password", "")) != password:
        return "wrong"
    expires_str = str(user.get("expires", ""))
    if expires_str:
        try:
            expires = datetime.strptime(expires_str, "%Y-%m-%d").date()
            if datetime.now().date() > expires:
                return "expired"
        except ValueError:
            pass
    return "ok"

def require_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if not st.session_state.authenticated:
        _screen_login()
        st.stop()

def _screen_login():
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 1.6, 1])
    with col_c:
        st.markdown("""
        <div style="background:rgba(11,11,28,0.9);border:1px solid #252545;border-radius:20px;
                    padding:40px 36px;backdrop-filter:blur(16px);box-shadow:0 0 60px rgba(127,86,217,0.15);">
            <div style="text-align:center;margin-bottom:24px;">
                <div style="font-size:52px;">🚀</div>
                <div style="font-size:28px;font-weight:900;background:linear-gradient(135deg,#a78bfa,#60a5fa);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:2px;">
                    KOSMOS AI</div>
                <div style="font-size:11px;color:#505070;letter-spacing:3px;text-transform:uppercase;margin-top:2px;">
                    Sales Training Simulator</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        login = st.text_input("Email / логин", placeholder="your@email.com", key="login_input")
        password = st.text_input("Пароль", type="password", placeholder="········", key="password_input")
        if st.button("Войти в систему →", use_container_width=True, type="primary"):
            if not login or not password:
                st.error("Введите логин и пароль.")
            else:
                res = _check_credentials(login, password)
                if res == "ok":
                    st.session_state.authenticated = True
                    st.session_state.current_user = login.strip().lower()
                    st.rerun()
                elif res == "expired":
                    st.error("⏰ Срок доступа истёк. Обратитесь к администратору.")
                elif res == "no_users":
                    st.session_state.authenticated = True
                    st.session_state.current_user = "dev"
                    st.rerun()
                else:
                    st.error("❌ Неверный логин или пароль.")

# ============================================================
#  SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        "screen": "menu",
        "persona_key": None,
        "mode": None,
        "messages": [],
        "stress": 0,
        "weak_streak": 0,
        "call_active": False,
        "call_start_time": None,
        "call_ended_reason": None,
        "review_result": None,
        "review_error": None,
        "deal_closed": False,
        "turn_count": 0,
        "auto_ended": False,
        "filter_topic": None,
        "filter_call_type": None,
        "last_tts_audio": None,
        "authenticated": False,
        "current_user": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()
require_auth()

# ============================================================
#  CSS — KOSMOS AI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif!important;background:#05050A!important;}
[data-testid="stAppViewContainer"]>.main{background:radial-gradient(ellipse at 20% 20%,#0d0d2b 0%,#05050A 60%)!important;}
.main .block-container{padding-top:1rem!important;padding-bottom:2rem!important;}
[data-testid="stSidebar"]{background:#0B0B16!important;border-right:1px solid #1e1e3a!important;}
[data-testid="stSidebar"] *{color:#c4c4e0!important;}
[data-testid="stSidebar"] .stButton>button{background:#131324!important;border:1px solid #252545!important;color:#c4c4e0!important;border-radius:10px!important;}
.stButton>button{font-family:'Inter',sans-serif!important;font-weight:600!important;border-radius:12px!important;transition:all 0.25s ease!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#7F56D9 0%,#5B3FBF 100%)!important;border:none!important;color:#fff!important;box-shadow:0 0 18px rgba(127,86,217,0.45)!important;}
.stButton>button[kind="secondary"]{background:#131324!important;border:1px solid #252545!important;color:#9090b8!important;}
.stButton>button[kind="secondary"]:hover{border-color:#7F56D9!important;color:#e0d7ff!important;}
h1,h2,h3,h4,h5{color:#e8e8ff!important;}p,li,label,.stMarkdown{color:#b0b0d0!important;}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:#0e0e20!important;border:1px solid #252545!important;border-radius:10px!important;color:#e0e0ff!important;}
.kosmos-card{background:rgba(15,15,36,0.85);backdrop-filter:blur(12px);border:1px solid #252545;border-radius:16px;padding:22px 18px 18px;margin-bottom:14px;transition:all 0.3s ease;position:relative;overflow:hidden;}
.kosmos-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:16px 16px 0 0;}
.kosmos-card.level-beginner{box-shadow:0 0 20px rgba(45,205,115,0.12);border-color:rgba(45,205,115,0.3);}
.kosmos-card.level-beginner::before{background:linear-gradient(90deg,transparent,#2DCD73,transparent);}
.kosmos-card.level-expert{box-shadow:0 0 20px rgba(45,140,255,0.12);border-color:rgba(45,140,255,0.3);}
.kosmos-card.level-expert::before{background:linear-gradient(90deg,transparent,#2D8CFF,transparent);}
.kosmos-card.level-hardcore{box-shadow:0 0 22px rgba(255,75,75,0.15);border-color:rgba(255,75,75,0.35);}
.kosmos-card.level-hardcore::before{background:linear-gradient(90deg,transparent,#FF4B4B,transparent);}
.kosmos-card:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(127,86,217,0.2);border-color:rgba(127,86,217,0.5);}
.kosmos-card-name{font-size:18px;font-weight:700;text-align:center;color:#e8e8ff;margin:10px 0 4px;}
.kosmos-card-badges{display:flex;justify-content:center;gap:6px;flex-wrap:wrap;margin:6px 0 10px;}
.kosmos-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.5px;}
.kosmos-card-tagline{font-size:12px;color:#7070a0;text-align:center;font-style:italic;margin-bottom:8px;line-height:1.4;}
.kosmos-card-desc{font-size:12px;color:#9090b8;line-height:1.55;margin-bottom:4px;}
.chat-row{display:flex;margin-bottom:10px;}
.chat-row.right{justify-content:flex-end;}
.chat-bubble-client{background:linear-gradient(135deg,#14143a 0%,#1a1a4a 100%);border:1px solid #2a2a5a;color:#d0d0f0;padding:10px 16px;border-radius:18px 18px 18px 4px;max-width:75%;font-size:14px;line-height:1.5;}
.chat-bubble-manager{background:linear-gradient(135deg,#2d1f5e 0%,#1e1542 100%);border:1px solid #4a3a80;color:#e8e0ff;padding:10px 16px;border-radius:18px 18px 4px 18px;max-width:75%;margin-left:auto;font-size:14px;line-height:1.5;}
.chat-name-label{font-size:11px;font-weight:600;color:#6060a0;margin-bottom:3px;}
.stress-label{font-size:12px;color:#606090;margin-bottom:5px;}
.stress-bar-track{background:#0e0e22;border-radius:8px;height:10px;overflow:hidden;border:1px solid #1e1e3a;}
.stress-bar-fill{height:100%;border-radius:8px;transition:width 0.5s ease;}
.score-badge{font-size:72px;font-weight:900;text-align:center;text-shadow:0 0 30px currentColor;line-height:1;}
.review-card{background:rgba(15,15,36,0.8);border:1px solid #252545;border-radius:14px;padding:18px;margin-bottom:12px;}
.kosmos-logo-title{font-size:32px;font-weight:900;background:linear-gradient(135deg,#a78bfa,#60a5fa,#a78bfa);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:2px;}
.kosmos-logo-sub{font-size:13px;color:#505070;letter-spacing:3px;text-transform:uppercase;margin-top:2px;}
/* Экран звонка */
.phone-screen{
    position:fixed;top:0;left:0;right:0;bottom:0;z-index:9999;
    background:radial-gradient(ellipse at top,#0a0a1e 0%,#000005 100%);
    display:flex;flex-direction:column;align-items:center;justify-content:space-between;
    padding:60px 20px 40px;
}
.phone-avatar{font-size:80px;margin-bottom:8px;text-align:center;}
.phone-name{font-size:28px;font-weight:800;color:#e8e8ff;text-align:center;}
.phone-status{font-size:13px;color:#606090;letter-spacing:2px;text-transform:uppercase;margin:6px 0;}
.phone-timer{font-size:42px;font-weight:900;color:#2DCD73;font-family:'Courier New',monospace;text-shadow:0 0 16px rgba(45,205,115,0.6);}
@keyframes pulse-dot{0%,100%{opacity:0.2;transform:scale(0.8);}50%{opacity:1;transform:scale(1.3);}}
.pulse-dots{display:flex;gap:8px;justify-content:center;margin:10px 0;}
.pulse-dot{width:10px;height:10px;border-radius:50%;background:#2DCD73;}
.phone-subtitles{width:100%;max-width:600px;margin-bottom:20px;}
.phone-subtitle-client{background:rgba(20,20,60,0.8);border:1px solid #2a2a5a;border-radius:12px 12px 12px 4px;padding:8px 14px;font-size:13px;color:#c0c0e0;margin-bottom:6px;}
.phone-subtitle-manager{background:rgba(45,31,94,0.8);border:1px solid #4a3a80;border-radius:12px 12px 4px 12px;padding:8px 14px;font-size:13px;color:#e0d8ff;text-align:right;}
.hangup-btn-wrap{display:flex;justify-content:center;}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  GIGACHAT — движок ИИ
# ============================================================

class AIClientError(Exception):
    pass


def _get_giga_client():
    """Инициализирует GigaChat клиент из ключа в secrets."""
    key = _get_sber_key()
    if not key:
        raise AIClientError("SBER_AUTH_KEY не найден в secrets.toml / переменных окружения.")
    if not GIGACHAT_AVAILABLE:
        raise AIClientError("Пакет gigachat не установлен. Выполните: pip install gigachat")
    return GigaChat(credentials=key, scope="GIGACHAT_API_PERS", verify_ssl_certs=False)


def _build_system_prompt(persona: Persona) -> str:
    return persona.system_prompt + _JSON_SUFFIX


def _parse_client_response(raw: str) -> Tuple[str, Optional[int]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
        text = str(data.get("response", "")).strip()
        raw_stress = data.get("stress_level")
        stress = int(raw_stress) if raw_stress is not None else None
        if stress is not None:
            stress = max(0, min(100, stress))
        if not text:
            return raw, None
        return text, stress
    except (json.JSONDecodeError, ValueError, TypeError):
        return raw, None


def call_ai_client(persona: Persona, history: List[Dict]) -> Tuple[str, Optional[int]]:
    """Вызывает GigaChat, возвращает (текст_реплики, stress_level)."""
    try:
        giga = _get_giga_client()
        msgs = [Messages(role=MessagesRole.SYSTEM, content=_build_system_prompt(persona))]
        for m in history:
            role = MessagesRole.USER if m["role"] == "manager" else MessagesRole.ASSISTANT
            msgs.append(Messages(role=role, content=m["text"]))
        resp = giga.chat(Chat(messages=msgs, model=GIGACHAT_MODEL, max_tokens=400, temperature=0.85))
        raw = resp.choices[0].message.content or ""
    except AIClientError:
        raise
    except Exception as e:
        raise AIClientError(f"Ошибка GigaChat API: {e}")
    return _parse_client_response(raw)

# ============================================================
#  SALUTESPEECH TTS
# ============================================================

def _get_salute_token(scope: str = "SALUTE_SPEECH_PERS") -> Optional[str]:
    """Получает access_token от SaluteSpeech."""
    try:
        resp = requests.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={
                "Authorization": _sber_auth_header(),
                "Content-Type":  "application/x-www-form-urlencoded",
                "RqUID":         str(uuid.uuid4()),
            },
            data={"scope": scope},
            verify=False,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception:
        return None


def tts_speak(text: str, persona_name: str = "") -> Optional[bytes]:
    """
    Синтезирует речь через SaluteSpeech.
    Строгие персонажи → эмоция strict; остальные → neutral.
    Возвращает байты MP3 или None при ошибке.
    """
    clean = re.sub(r'\*[^*]+\*', '', text).strip()
    if not clean:
        return None

    token = _get_salute_token()
    if not token:
        return None

    emotion = "strict" if persona_name in STRICT_VOICES else "neutral"

    try:
        resp = requests.post(
            "https://smartspeech.sber.ru/rest/v1/text:synthesize",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/text",
            },
            params={
                "format":   "mp3",
                "language": "ru-RU",
                "emotion":  emotion,
            },
            data=clean.encode("utf-8"),
            verify=False,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None

# ============================================================
#  СТРЕСС / АВТО-ЗАВЕРШЕНИЕ
# ============================================================

WEAK_MARKERS = [
    "не знаю","наверное","может быть","извините","простите",
    "не уверен","ну как бы","не могу сказать точно",
    "скидку дам","давайте скину","хорошо, скидка","ладно, скидка",
]

def detect_weak_phrase(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in WEAK_MARKERS)

def update_stress(manager_text: str):
    if detect_weak_phrase(manager_text):
        st.session_state.weak_streak += 1
        st.session_state.stress = min(100, st.session_state.stress + random.randint(12, 22))
    else:
        st.session_state.weak_streak = 0
        if len(manager_text.strip()) > 25:
            st.session_state.stress = max(0, st.session_state.stress - random.randint(3, 10))
        else:
            st.session_state.stress = min(100, st.session_state.stress + random.randint(2, 6))

def check_auto_end(persona: Persona) -> bool:
    return st.session_state.stress >= 100 or st.session_state.weak_streak >= persona.patience

def trigger_auto_end(persona: Persona):
    st.session_state.messages.append({
        "role": "client",
        "text": f"*{persona.name} {'повесил трубку' if st.session_state.mode == 'call' else 'прервал чат'}* — {persona.hangup_line}",
        "ts": datetime.now().strftime("%H:%M"),
    })
    st.session_state.call_active = False
    st.session_state.call_ended_reason = "auto_hangup"
    st.session_state.auto_ended = True
    st.session_state.deal_closed = True
    st.session_state.screen = "review"

# ============================================================
#  СТРЕСС-БАР
# ============================================================

def get_persona() -> Optional[Persona]:
    if st.session_state.persona_key:
        return CLIENTS_DB.get(st.session_state.persona_key)
    return None

def reset_dialog_state():
    st.session_state.messages = []
    st.session_state.stress = 0
    st.session_state.weak_streak = 0
    st.session_state.call_active = False
    st.session_state.call_start_time = None
    st.session_state.call_ended_reason = None
    st.session_state.review_result = None
    st.session_state.review_error = None
    st.session_state.deal_closed = False
    st.session_state.turn_count = 0
    st.session_state.auto_ended = False
    st.session_state.last_tts_audio = None

def render_stress_bar(persona: Persona):
    v = st.session_state.stress
    if v < 35:   color, label, glow = "#2DCD73","Спокоен","rgba(45,205,115,0.4)"
    elif v < 70: color, label, glow = "#F5A623","Раздражён","rgba(245,166,35,0.4)"
    else:        color, label, glow = "#FF4B4B","На грани!","rgba(255,75,75,0.5)"
    st.markdown(f"""
    <div class="stress-label">😤 Стресс «{persona.name}»:
        <b style="color:{color};text-shadow:0 0 8px {glow};">{label}</b>
        <span style="color:#404060;font-size:11px;margin-left:6px;">{v}%</span>
    </div>
    <div class="stress-bar-track">
        <div class="stress-bar-fill" style="width:{v}%;background:linear-gradient(90deg,{color}88,{color});box-shadow:0 0 8px {glow};"></div>
    </div>""", unsafe_allow_html=True)

# ============================================================
#  ОТПРАВКА РЕПЛИКИ МЕНЕДЖЕРА
# ============================================================

def manager_send(text: str, persona: Persona) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    st.session_state.messages.append({"role":"manager","text":text,"ts":datetime.now().strftime("%H:%M")})
    update_stress(text)
    if check_auto_end(persona):
        trigger_auto_end(persona)
        return None
    try:
        with st.spinner(f"{persona.name} отвечает..."):
            reply, ai_stress = call_ai_client(persona, st.session_state.messages)
    except AIClientError as e:
        return str(e)
    if ai_stress is not None:
        st.session_state.stress = ai_stress
        if ai_stress >= 75:
            st.session_state.weak_streak += 1
        else:
            st.session_state.weak_streak = 0
    st.session_state.messages.append({"role":"client","text":reply,"ts":datetime.now().strftime("%H:%M")})
    st.session_state.turn_count += 1
    audio = tts_speak(reply, persona.name)
    st.session_state.last_tts_audio = audio
    if check_auto_end(persona):
        trigger_auto_end(persona)
    return None

# ============================================================
#  AI-СУДЬЯ
# ============================================================

JUDGE_SYSTEM_PROMPT = """Ты — опытный директор отдела продаж и бизнес-тренер.
Тебе дают полный лог переговоров менеджера с ИИ-клиентом.
Проанализируй: удержание инициативы, работу с возражениями (особенно цена), уверенность позиции, выявление потребностей, закрытие сделки.
Верни СТРОГО валидный JSON без markdown:
{
  "score": <1-10>,
  "summary": "<2-3 предложения>",
  "strengths": ["<сила 1>","<сила 2>"],
  "mistakes": [{"moment":"<момент>","issue":"<проблема>","fix":"<как надо>"}],
  "price_handling": "<комментарий по цене>",
  "next_steps": ["<рекомендация 1>","<рекомендация 2>"]
}"""

def _build_transcript(persona: Persona) -> str:
    lines = []
    for m in st.session_state.messages:
        speaker = "МЕНЕДЖЕР" if m["role"] == "manager" else persona.name.upper()
        lines.append(f"{speaker}: {m['text']}")
    return "\n".join(lines)

def _try_repair_json(cleaned: str) -> Optional[Dict]:
    s = cleaned
    if s.count('"') % 2 == 1:
        s += '"'
    stack = []
    in_string = False
    escape = False
    for ch in s:
        if escape:   escape = False; continue
        if ch == "\\": escape = True; continue
        if ch == '"':  in_string = not in_string; continue
        if in_string: continue
        if ch in "{[": stack.append(ch)
        elif ch in "}]" and stack: stack.pop()
    closers = {"{":"}","[":"]"}
    while stack:
        s += closers[stack.pop()]
    try:    return json.loads(s)
    except: return None

def _parse_judge_json(raw: str) -> Dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"): cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        if start != -1:
            end = cleaned.rfind("}")
            cleaned = cleaned[start:end+1] if end > start else cleaned[start:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        rep = _try_repair_json(cleaned)
        if rep: return rep
        raise

def run_ai_judge(persona: Persona) -> Dict:
    transcript = _build_transcript(persona)
    user_prompt = f"""ПЕРСОНАЖ: {persona.name} ({persona.level})
ОПИСАНИЕ: {persona.description}
ЛОГ ДИАЛОГА:
{transcript}
Проанализируй работу менеджера и верни JSON."""
    try:
        giga = _get_giga_client()
        msgs = [
            Messages(role=MessagesRole.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
            Messages(role=MessagesRole.USER,   content=user_prompt),
        ]
        resp = giga.chat(Chat(messages=msgs, model=GIGACHAT_MODEL, max_tokens=2000, temperature=0.3))
        raw = resp.choices[0].message.content or ""
    except AIClientError:
        raise
    except Exception as e:
        raise AIClientError(f"Ошибка GigaChat при оценке: {e}")
    if not raw.strip():
        raise AIClientError("GigaChat вернул пустой ответ при оценке.")
    try:
        result = _parse_judge_json(raw)
        result["_partial"] = False
        return result
    except json.JSONDecodeError as e:
        raise AIClientError(f"Не удалось разобрать JSON оценки: {e}\n\nОтвет: {raw[:400]}")

# ============================================================
#  БОКОВАЯ ПАНЕЛЬ
# ============================================================

with st.sidebar:
    user = st.session_state.get("current_user", "")
    users = _get_users()
    if user and user in users:
        expires_str = str(users[user].get("expires", ""))
        try:
            expires = datetime.strptime(expires_str, "%Y-%m-%d").date()
            days_left = (expires - datetime.now().date()).days
            if days_left <= 3:
                st.warning(f"⏰ Доступ истекает через {days_left} дн. ({expires_str})")
            else:
                st.caption(f"👤 {user} · доступ до {expires_str}")
        except Exception:
            st.caption(f"👤 {user}")
    elif user == "dev":
        st.caption("👤 dev-режим")

    if st.button("🚪 Выйти", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()

    st.divider()
    st.markdown("### 🤖 ИИ-движок")
    _key_ok = bool(_get_sber_key()) and GIGACHAT_AVAILABLE
    if _key_ok:
        st.markdown(f"""
        <div style="background:rgba(45,205,115,0.1);border:1px solid rgba(45,205,115,0.3);
                    border-radius:10px;padding:10px 14px;">
            <div style="color:#2DCD73;font-weight:700;font-size:13px;">✅ GigaChat-Pro подключён</div>
            <div style="color:#505070;font-size:12px;margin-top:2px;">Сбер · {GIGACHAT_MODEL}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,75,75,0.1);border:1px solid rgba(255,75,75,0.3);
                    border-radius:10px;padding:10px 14px;">
            <div style="color:#FF4B4B;font-weight:700;font-size:13px;">⚠️ Ключ не найден</div>
            <div style="color:#505070;font-size:12px;margin-top:2px;">Добавьте SBER_AUTH_KEY в secrets.toml</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Текущая сессия")
    if get_persona():
        p = get_persona()
        st.markdown(f"**Персонаж:** {p.emoji} {p.name} ({p.level})")
        st.markdown(f"**Режим:** {'💬 Чат' if st.session_state.mode == 'chat' else '📞 Звонок'}")
        st.markdown(f"**Реплик:** {len(st.session_state.messages)}")
        st.markdown(f"**Слабых подряд:** {st.session_state.weak_streak} / {p.patience}")
    else:
        st.markdown("_Персонаж не выбран_")

    st.divider()
    if st.button("🔄 В меню", use_container_width=True):
        st.session_state.screen = "menu"
        st.session_state.persona_key = None
        st.session_state.mode = None
        reset_dialog_state()
        st.rerun()

# ============================================================
#  ЭКРАН ТЕЛЕФОННОГО ЗВОНКА
# ============================================================

def screen_phone_call():
    """Экран звонка: таймер, субтитры, ввод, кнопки управления."""
    persona = get_persona()
    if not persona:
        st.session_state.call_active = False
        st.rerun()
        return

    # ── Первая реплика персонажа ────────────────────────────
    if not st.session_state.messages:
        opening = random.choice(persona.opening_lines)
        st.session_state.messages.append({
            "role": "client", "text": opening, "ts": datetime.now().strftime("%H:%M")
        })
        audio = tts_speak(opening, persona.name)
        if audio:
            st.audio(audio, format="audio/mp3", autoplay=True)

    # ── Таймер ──────────────────────────────────────────────
    start = st.session_state.call_start_time or time.time()
    elapsed = int(time.time() - start)
    mins, secs = divmod(elapsed, 60)

    # ── Субтитры — последние реплики ────────────────────────
    last_client  = next((m["text"] for m in reversed(st.session_state.messages)
                         if m["role"] == "client"), "")
    last_manager = next((m["text"] for m in reversed(st.session_state.messages)
                         if m["role"] == "manager"), "")

    def _clip(t: str, n: int = 110) -> str:
        return t[:n] + "…" if len(t) > n else t

    # ── Аватар ──────────────────────────────────────────────
    if persona.avatar_url:
        avatar_html = (
            f'<img src="{persona.avatar_url}" '
            f'style="width:96px;height:96px;border-radius:50%;object-fit:cover;'
            f'border:3px solid {persona.level_color}88;display:block;margin:0 auto 10px auto;">'
        )
    else:
        avatar_html = (
            f'<div style="font-size:72px;text-align:center;margin-bottom:10px;">'
            f'{persona.emoji}</div>'
        )

    # ── Субтитры HTML (только если есть текст) ──────────────
    subtitle_client_html = (
        f'<div style="background:rgba(20,20,60,0.85);border:1px solid #2a2a5a;'
        f'border-radius:12px 12px 12px 4px;padding:9px 14px;'
        f'font-size:13px;color:#c0c0e0;margin-bottom:6px;">'
        f'{persona.emoji} {_clip(last_client)}</div>'
    ) if last_client else ""

    subtitle_manager_html = (
        f'<div style="background:rgba(45,31,94,0.85);border:1px solid #4a3a80;'
        f'border-radius:12px 12px 4px 12px;padding:9px 14px;'
        f'font-size:13px;color:#e0d8ff;text-align:right;">'
        f'👤 {_clip(last_manager)}</div>'
    ) if last_manager else ""

    # ── Основной HTML-блок — всё закрыто внутри одного st.markdown ──
    st.markdown(f"""
    <div style="
        background: radial-gradient(ellipse at top, #0a0a1e 0%, #000005 100%);
        border-radius: 24px;
        border: 1px solid #1a1a3a;
        padding: 36px 24px 28px 24px;
        text-align: center;
        box-shadow: 0 0 60px rgba(45,205,115,0.08);
        margin-bottom: 16px;
    ">
        {avatar_html}
        <div style="font-size:26px;font-weight:800;color:#e8e8ff;margin-bottom:4px;">
            {persona.name}
        </div>
        <div style="font-size:12px;color:#606090;letter-spacing:2px;
                    text-transform:uppercase;margin-bottom:8px;">
            📞 &nbsp; ИДЁт ЗВОНОК
        </div>
        <div style="font-size:44px;font-weight:900;color:#2DCD73;
                    font-family:'Courier New',monospace;
                    text-shadow:0 0 16px rgba(45,205,115,0.6);margin-bottom:10px;">
            {mins:02d}:{secs:02d}
        </div>
        <div style="display:flex;gap:8px;justify-content:center;margin-bottom:20px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#2DCD73;
                        animation:pulse-dot 1.2s infinite 0s;"></div>
            <div style="width:10px;height:10px;border-radius:50%;background:#2DCD73;
                        animation:pulse-dot 1.2s infinite 0.4s;"></div>
            <div style="width:10px;height:10px;border-radius:50%;background:#2DCD73;
                        animation:pulse-dot 1.2s infinite 0.8s;"></div>
        </div>
        <div style="text-align:left;max-width:560px;margin:0 auto;">
            {subtitle_client_html}
            {subtitle_manager_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Аудио TTS ───────────────────────────────────────────
    if st.session_state.get("last_tts_audio"):
        st.audio(st.session_state.last_tts_audio, format="audio/mp3", autoplay=True)
        st.session_state.last_tts_audio = None

    # ── Стресс-бар ──────────────────────────────────────────
    render_stress_bar(persona)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Голосовой / текстовый ввод ──────────────────────────
    if MIC_AVAILABLE:
        voice_text = speech_to_text(
            language="ru",
            start_prompt="🎤 Говорить",
            stop_prompt="🛑 Отправить",
            just_once=True,
            key="voice_call",
        )
        if voice_text:
            err = manager_send(voice_text, persona)
            if err:
                st.error(f"Ошибка ИИ: {err}")
            if st.session_state.get("screen") == "review":
                st.session_state.call_active = False
            st.rerun()
    else:
        with st.form("call_text_form", clear_on_submit=True):
            col_i, col_b = st.columns([5, 1])
            with col_i:
                txt = st.text_input(
                    "Реплика", label_visibility="collapsed",
                    placeholder="Напишите вашу реплику..."
                )
            with col_b:
                sent = st.form_submit_button("➤", use_container_width=True, type="primary")
        if sent and txt:
            err = manager_send(txt, persona)
            if err:
                st.error(f"Ошибка ИИ: {err}")
            if st.session_state.get("screen") == "review":
                st.session_state.call_active = False
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Кнопки управления ───────────────────────────────────
    col_hang, col_menu = st.columns(2)

    with col_hang:
        if st.button("❌  Положить трубку", use_container_width=True, type="primary"):
            st.session_state.call_active = False
            st.session_state.call_ended_reason = "manager_ended"
            st.session_state.deal_closed = True
            # Вызываем судью сразу здесь, чтобы review-экран не ждал спиннера
            if st.session_state.messages and any(
                m["role"] == "manager" for m in st.session_state.messages
            ):
                try:
                    with st.spinner("ИИ-судья анализирует звонок..."):
                        st.session_state.review_result = run_ai_judge(persona)
                    save_to_csv(persona, st.session_state.review_result)
                except Exception as e:
                    st.session_state.review_error = str(e)
            st.session_state.screen = "review"
            st.rerun()

    with col_menu:
        if st.button("⬅️  В главное меню", use_container_width=True, type="secondary"):
            st.session_state.call_active = False
            reset_dialog_state()
            st.session_state.screen = "menu"
            st.rerun()

    # ── Тикаем таймер каждую секунду ────────────────────────
    # Делаем это только если звонок ещё активен и диалог не завершён
    if st.session_state.get("call_active") and st.session_state.get("screen") != "review":
        time.sleep(1)
        st.rerun()

# ============================================================
#  ЭКРАН МЕНЮ
# ============================================================

def screen_menu():
    INSOFT_TOPIC = "Insoft: Холодные продажи B2B"

    st.markdown(f"""
    <div style="text-align:center;margin-bottom:16px;">
        <div class="kosmos-logo-title">🚀 KOSMOS AI</div>
        <div class="kosmos-logo-sub">Sales Training Simulator</div>
        <div style="margin-top:6px;display:inline-block;
                    background:linear-gradient(135deg,#7F56D9,#60a5fa);
                    padding:3px 14px;border-radius:20px;font-size:11px;font-weight:700;
                    letter-spacing:2px;color:#fff;">✦ SPECIAL EDITION FOR INSOFT ✦</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Шаг 1 — Выберите тему")
    other_topics = sorted({p.topic for p in CLIENTS_DB.values() if p.topic != INSOFT_TOPIC})
    all_topics = [INSOFT_TOPIC] + other_topics
    if not st.session_state.get("filter_topic"):
        st.session_state.filter_topic = INSOFT_TOPIC

    topic_cols = st.columns(len(all_topics))
    for col, topic in zip(topic_cols, all_topics):
        with col:
            icon = "🎯" if topic == INSOFT_TOPIC else TOPIC_ICON.get(topic, "📁")
            selected = st.session_state.get("filter_topic") == topic
            if st.button(f"{icon} {topic}" + (" ✓" if selected else ""),
                         key=f"topic_{topic}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                st.session_state.filter_topic = topic
                st.session_state.filter_call_type = None
                st.rerun()

    chosen_topic = st.session_state.get("filter_topic")
    if not chosen_topic:
        st.info("👆 Выберите тему.")
        return

    st.markdown(f"### Шаг 2 — Тип звонка")
    if chosen_topic == INSOFT_TOPIC:
        insoft_types = ["Холодный B2B", "Холодный B2C"]
        ct_cols = st.columns(2)
        for col, ct in zip(ct_cols, insoft_types):
            with col:
                selected = st.session_state.get("filter_call_type") == ct
                if st.button(f"❄️ Холодный звонок ({ct.split()[-1]})" + (" ✓" if selected else ""),
                             key=f"ct_insoft_{ct}", use_container_width=True,
                             type="primary" if selected else "secondary"):
                    st.session_state.filter_call_type = ct
                    st.rerun()
    else:
        call_types = sorted({p.call_type for p in CLIENTS_DB.values() if p.topic == chosen_topic})
        ct_cols = st.columns(max(len(call_types), 1))
        for col, ct in zip(ct_cols, call_types):
            with col:
                icon = CALL_TYPE_ICON.get(ct, "📞")
                selected = st.session_state.get("filter_call_type") == ct
                if st.button(f"{icon} {ct} звонок" + (" ✓" if selected else ""),
                             key=f"ct_{ct}", use_container_width=True,
                             type="primary" if selected else "secondary"):
                    st.session_state.filter_call_type = ct
                    st.rerun()

    chosen_ct = st.session_state.get("filter_call_type")
    if not chosen_ct:
        st.info("👆 Выберите тип звонка.")
        return

    filtered = [p for p in CLIENTS_DB.values()
                if p.topic == chosen_topic and p.call_type == chosen_ct]
    if not filtered:
        st.warning("Клиенты с такими параметрами не найдены.")
        return

    st.markdown("### Шаг 3 — Выберите клиента")
    cols = st.columns(min(len(filtered), 3))
    for col, persona in zip(cols, filtered):
        with col:
            level_class = {"Новичок":"level-beginner","Опытный":"level-expert","Хардкор":"level-hardcore"}.get(persona.level,"level-expert")
            avatar_html = (
                f'<img src="{persona.avatar_url}" style="width:76px;height:76px;border-radius:50%;object-fit:cover;display:block;margin:0 auto 10px auto;border:2px solid {persona.level_color}44;">'
                if persona.avatar_url else
                f'<div style="font-size:60px;text-align:center;margin-bottom:10px;">{persona.emoji}</div>'
            )
            ct_icon = CALL_TYPE_ICON.get(persona.call_type, "📞")
            st.markdown(f"""
            <div class="kosmos-card {level_class}">
                {avatar_html}
                <div class="kosmos-card-name">{persona.name}</div>
                <div class="kosmos-card-badges">
                    <span class="kosmos-badge" style="background:{persona.level_color}22;color:{persona.level_color};border:1px solid {persona.level_color}55;">{persona.level.upper()}</span>
                    <span class="kosmos-badge" style="background:#1a1a3a;color:#6060a0;border:1px solid #252545;">{ct_icon} {persona.call_type}</span>
                </div>
                <div class="kosmos-card-tagline">{persona.tagline}</div>
                <div class="kosmos-card-desc">{persona.description}</div>
            </div>""", unsafe_allow_html=True)

            col_chat, col_voice = st.columns(2)
            with col_chat:
                if st.button("💬 Чат", key=f"chat_{persona.key}", use_container_width=True, type="secondary"):
                    st.session_state.persona_key = persona.key
                    st.session_state.mode = "chat"
                    reset_dialog_state()
                    st.session_state.stress = persona.stress_start
                    st.session_state.screen = "chat"
                    st.rerun()
            with col_voice:
                if st.button("📞 Голосовой", key=f"voice_{persona.key}", use_container_width=True, type="primary"):
                    st.session_state.persona_key = persona.key
                    st.session_state.mode = "call"
                    reset_dialog_state()
                    st.session_state.stress = persona.stress_start
                    st.session_state.call_active = True
                    st.session_state.call_start_time = time.time()
                    st.session_state.screen = "call"
                    st.rerun()

# ============================================================
#  ЭКРАН ТЕКСТОВОГО ЧАТА
# ============================================================

def screen_chat():
    persona = get_persona()
    if not persona:
        st.session_state.screen = "menu"
        st.rerun()
        return

    col_l, col_r = st.columns([4,1])
    with col_l:
        ct_icon = CALL_TYPE_ICON.get(persona.call_type, "📞")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"<span style='font-size:28px;'>{persona.emoji}</span>"
            f"<div><div style='font-size:20px;font-weight:800;color:#e8e8ff;'>{persona.name}</div>"
            f"<div style='font-size:12px;color:#505070;'>"
            f"<span style='color:{persona.level_color};font-weight:700;'>{persona.level}</span>"
            f" · {ct_icon} {persona.call_type} · {persona.topic}</div></div></div>",
            unsafe_allow_html=True)
    with col_r:
        if st.button("← Меню", use_container_width=True):
            st.session_state.screen = "menu"
            st.rerun()

    render_stress_bar(persona)
    st.divider()

    if not st.session_state.messages:
        opening = random.choice(persona.opening_lines)
        st.session_state.messages.append({"role":"client","text":opening,"ts":datetime.now().strftime("%H:%M")})

    chat_container = st.container(height=420)
    with chat_container:
        for m in st.session_state.messages:
            if m["role"] == "client":
                st.markdown(f"""<div class="chat-row">
                    <div><div class="chat-name-label">{persona.emoji} {persona.name}</div>
                    <div class="chat-bubble-client">{m['text']}</div>
                    <div style="font-size:10px;color:#404060;margin-top:3px;">{m['ts']}</div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-row right">
                    <div><div class="chat-name-label" style="text-align:right;color:#5050a0;">👤 Вы</div>
                    <div class="chat-bubble-manager">{m['text']}</div>
                    <div style="font-size:10px;color:#404060;margin-top:3px;text-align:right;">{m['ts']}</div></div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    if st.session_state.get("last_tts_audio"):
        st.audio(st.session_state.last_tts_audio, format="audio/mp3", autoplay=True)
        st.session_state.last_tts_audio = None

    if st.session_state.auto_ended:
        st.error(f"💬 {persona.name} прервал чат.")
        st.session_state.screen = "review"
        st.rerun()
        return

    if not st.session_state.deal_closed:
        with st.form("chat_form", clear_on_submit=True):
            col_i, col_b = st.columns([5,1])
            with col_i:
                user_text = st.text_input("Сообщение", placeholder="Напишите ответ клиенту...", label_visibility="collapsed")
            with col_b:
                submitted = st.form_submit_button("➤", use_container_width=True, type="primary")
        if submitted and user_text:
            err = manager_send(user_text, persona)
            if err: st.error(f"Ошибка: {err}")
            st.rerun()
        st.markdown("")
        if st.button("🏁 Завершить сделку и получить оценку", use_container_width=True, type="secondary"):
            st.session_state.deal_closed = True
            st.session_state.screen = "review"
            st.rerun()
    else:
        st.info("Диалог завершён. Переход к разбору полётов...")
        st.session_state.screen = "review"
        st.rerun()

# ============================================================
#  ЭКРАН РАЗБОРА ПОЛЁТОВ
# ============================================================

def screen_review():
    persona = get_persona()
    if not persona:
        st.session_state.screen = "menu"
        st.rerun()
        return

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <div style="font-size:32px;">🧑‍⚖️</div>
        <div>
            <div style="font-size:22px;font-weight:800;color:#e8e8ff;">Разбор полётов</div>
            <div style="font-size:13px;color:#505070;">
                Переговоры с {persona.emoji} {persona.name} · {persona.topic} · {persona.call_type}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.auto_ended:
        reason = "клиент достиг макс. стресса" if st.session_state.stress >= 100 else "слишком много слабых ответов"
        st.warning(f"⚠️ Диалог автоматически завершён клиентом ({reason}).")

    manager_turns = [m for m in st.session_state.messages if m["role"] == "manager"]
    if not st.session_state.messages or len(manager_turns) == 0:
        st.warning("Диалог слишком короткий для анализа.")
        if st.button("← Вернуться к диалогу", key="back_short_dialog"):
            st.session_state.screen = "chat" if st.session_state.mode == "chat" else "call"
            st.session_state.deal_closed = False
            st.session_state.auto_ended = False
            st.session_state.review_result = None
            st.session_state.review_error = None
            st.rerun()
        return

    if not _get_sber_key() or not GIGACHAT_AVAILABLE:
        st.error("SBER_AUTH_KEY не настроен — оценка недоступна.")
        if st.button("← Вернуться к диалогу", key="back_no_api"):
            st.session_state.screen = "chat" if st.session_state.mode == "chat" else "call"
            st.session_state.deal_closed = False
            st.rerun()
        return

    if st.session_state.review_result is None and st.session_state.review_error is None:
        with st.spinner("ИИ-судья анализирует переговоры..."):
            try:
                st.session_state.review_result = run_ai_judge(persona)
            except AIClientError as e:
                st.session_state.review_error = str(e)
            except Exception as e:
                st.session_state.review_error = f"Неожиданная ошибка: {type(e).__name__}: {e}"

    if st.session_state.review_error:
        err = st.session_state.review_error
        if "404" in err or "not found" in err.lower():
            st.error(f"❌ Модель не найдена (404): `{err}`")
        elif "429" in err or "quota" in err.lower():
            st.error(f"❌ Превышена квота GigaChat: `{err}`")
        else:
            st.error(f"❌ Ошибка оценки:\n\n`{err}`")
        col_r, col_b = st.columns(2)
        with col_r:
            if st.button("🔁 Повторить", use_container_width=True, type="primary"):
                st.session_state.review_error = None
                st.rerun()
        with col_b:
            if st.button("← Вернуться к диалогу", use_container_width=True, key="back_api_error"):
                st.session_state.screen = "chat" if st.session_state.mode == "chat" else "call"
                st.session_state.deal_closed = False
                st.session_state.review_result = None
                st.session_state.review_error = None
                st.rerun()
        return

    result = st.session_state.review_result
    if result.get("_partial"):
        st.warning("⚠️ Ответ модели был обрезан — результат частичный.")
        if st.button("🔁 Пересчитать", key="recompute_partial"):
            st.session_state.review_result = None
            st.session_state.review_error = None
            st.rerun()

    score = int(result.get("score", 5))
    score = max(1, min(10, score))
    if score >= 8:   sc, sg = "#2DCD73","rgba(45,205,115,0.5)"; sl = "Отличный результат"
    elif score >= 5: sc, sg = "#F5A623","rgba(245,166,35,0.4)"; sl = "Есть над чем поработать"
    else:            sc, sg = "#FF4B4B","rgba(255,75,75,0.5)";   sl = "Нужна серьёзная работа"

    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown(f"""
        <div style="text-align:center;background:rgba(15,15,36,0.8);border-radius:16px;
                    padding:24px 16px;border:1px solid {sc}44;box-shadow:0 0 30px {sg};">
            <div class="score-badge" style="color:{sc};text-shadow:0 0 24px {sg};">{score}</div>
            <div style="font-size:11px;color:{sc};font-weight:700;margin-top:2px;">/ 10</div>
            <div style="color:#505070;font-size:12px;margin-top:8px;">{sl}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="review-card">
            <div style="font-size:13px;font-weight:700;color:#7070a0;letter-spacing:1px;margin-bottom:8px;">📝 ОБЩИЙ ВЫВОД</div>
            <div style="color:#c0c0e0;line-height:1.6;font-size:14px;">{result.get("summary","—")}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_s, col_m = st.columns(2)
    with col_s:
        st.markdown(f"""<div class="review-card" style="border-color:rgba(45,205,115,0.3);">
            <div style="font-size:13px;font-weight:700;color:#2DCD73;letter-spacing:1px;margin-bottom:10px;">✅ СИЛЬНЫЕ СТОРОНЫ</div>
        </div>""", unsafe_allow_html=True)
        for s in result.get("strengths", []):
            st.markdown(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_m:
        st.markdown(f"""<div class="review-card" style="border-color:rgba(255,75,75,0.25);">
            <div style="font-size:13px;font-weight:700;color:#FF4B4B;letter-spacing:1px;margin-bottom:10px;">❌ ОШИБКИ</div>
        </div>""", unsafe_allow_html=True)
        for mistake in result.get("mistakes", []):
            with st.expander(f"⚠️ {mistake.get('moment','Момент')}"):
                st.markdown(f"**Проблема:** {mistake.get('issue','')}")
                st.markdown(f"**Как надо:** {mistake.get('fix','')}")
        st.markdown("</div>", unsafe_allow_html=True)

    price_text = result.get("price_handling","—")
    st.markdown(f"""
    <div class="review-card" style="border-color:rgba(127,86,217,0.3);margin-top:8px;">
        <div style="font-size:13px;font-weight:700;color:#7F56D9;letter-spacing:1px;margin-bottom:8px;">💰 РАБОТА С ЦЕНОЙ</div>
        <div style="color:#c0c0e0;font-size:14px;line-height:1.6;">{price_text}</div>
    </div>""", unsafe_allow_html=True)

    next_steps = result.get("next_steps", [])
    if next_steps:
        steps_html = "".join(
            f'<div style="display:flex;gap:10px;margin-bottom:8px;">'
            f'<span style="color:#7F56D9;font-weight:700;">→</span>'
            f'<span style="color:#c0c0e0;font-size:14px;">{s}</span></div>'
            for s in next_steps
        )
        st.markdown(f"""
        <div class="review-card" style="margin-top:8px;">
            <div style="font-size:13px;font-weight:700;color:#60a5fa;letter-spacing:1px;margin-bottom:10px;">🚀 РЕКОМЕНДАЦИИ</div>
            {steps_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.divider()

    # CSV сохранение
    save_key = f"csv_saved_{id(result)}"
    if not st.session_state.get(save_key):
        saved = save_to_csv(persona, result)
        st.session_state[save_key] = True
        if saved:
            st.success("💾 Результат сохранён в `insoft_history.csv`")

    if HISTORY_CSV.exists():
        with open(HISTORY_CSV, "rb") as f:
            st.download_button("⬇️ Скачать историю (CSV)", data=f.read(),
                               file_name="insoft_history.csv", mime="text/csv",
                               use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔁 Попробовать снова", use_container_width=True):
            reset_dialog_state()
            st.session_state.stress = persona.stress_start
            st.session_state.screen = "chat" if st.session_state.mode == "chat" else "call"
            st.rerun()
    with col_b:
        if st.button("← Выбрать другого клиента", use_container_width=True, type="primary"):
            st.session_state.screen = "menu"
            st.session_state.persona_key = None
            reset_dialog_state()
            st.rerun()

    with st.expander("📜 Полный лог диалога"):
        for m in st.session_state.messages:
            who = "👤 Менеджер" if m["role"] == "manager" else f"{persona.emoji} {persona.name}"
            st.markdown(f"**{who}:** {m['text']}")

# ============================================================
#  РОУТЕР
# ============================================================

# Если идёт голосовой звонок — показываем только экран звонка
if st.session_state.get("call_active"):
    screen_phone_call()
else:
    screen = st.session_state.screen
    if screen == "menu":
        screen_menu()
    elif screen == "chat":
        screen_chat()
    elif screen == "review":
        screen_review()
    else:
        st.session_state.screen = "menu"
        st.rerun()
