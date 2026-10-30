export interface FeatureItem {
  id: string;
  name: string;
  vision: boolean;
  isCameraTracker?: boolean;
  isGeneralChat?: boolean;
  isFlowchart?: boolean;
  isStudyPlanner?: boolean;
  isQuiz?: boolean;
  isSpamAudit?: boolean;
  isCropWeather?: boolean;
  desc: string;
  defaultInstruction?: string;
  systemPrompt: string;
}

export interface CategoryGroup {
  category: string;
  items: FeatureItem[];
}

export const CATEGORIES_HIERARCHY: CategoryGroup[] = [
  {
    category: '✨ QUICKIO CORE',
    items: [
      {
        id: 'general-chat',
        name: 'Quickio Pulse',
        vision: false,
        isGeneralChat: true,
        desc: 'Your central multimodal AI companion with persistent chat history.',
        systemPrompt:
          'You are Quickio Pulse, a helpful, accurate, multimodal AI assistant inside Quickio AI. Answer questions clearly, analyze any attached photos, and assist with coding, writing, analysis, and daily tasks.',
      },
    ],
  },
  {
    category: '👁️ VISION & ECOLOGY',
    items: [
      {
        id: 'waste-sorter',
        name: 'Waste Sorter',
        vision: true,
        desc: 'Classify waste items into Recyclable, Compostable, E-Waste, or Landfill with disposal instructions.',
        systemPrompt:
          'You are Quickio AI Waste Sorter. Analyze the uploaded or webcam-captured photo or item description. Identify the material composition, assign the proper bin category (Recyclable, Organic/Compost, Hazardous/E-Waste, or General Waste), and give step-by-step preparation tips.',
      },
      {
        id: 'plant-species-identifier',
        name: 'Plant Species Identifier',
        vision: true,
        desc: 'Identify plant species, botanical taxonomy, care requirements, and leaf health from photos.',
        systemPrompt:
          'You are Quickio AI Plant Species Identifier & Botanist. Identify the exact plant species (common name, scientific name, and family), growth habit, sunlight/watering needs, toxicity to pets, and any visible leaf health issues.',
      },
      {
        id: 'fruit-freshness',
        name: 'Fruit Freshness',
        vision: true,
        desc: 'Evaluate ripeness index, surface bruising, mold indicators, and remaining shelf life.',
        systemPrompt:
          'You are Quickio AI Fruit Freshness Inspector. Assess the visual ripeness stage, peel integrity, oxidation/bruising, estimated shelf life in days, and storage recommendations.',
      },
      {
        id: 'live-camera-tracker',
        name: 'Live Camera Bounding Box Tracker',
        vision: true,
        isCameraTracker: true,
        desc: 'Quickio Live real-time camera & voice mode with instant 30 FPS on-device multi-object bounding box tracking.',
        systemPrompt:
          'You are Quickio AI in Quickio Live Vision mode. You are looking directly at the user live webcam feed. Answer their spoken or typed questions naturally and concisely based on what is visible in their camera frame, and identify objects in view.',
      },
    ],
  },
  {
    category: '🛡️ SAFETY & BIOMETRICS',
    items: [
      {
        id: 'posture-ergonomics',
        name: 'Posture Ergonomics',
        vision: true,
        desc: 'Assess desk seating angles, cervical spine alignment, and wrist positioning.',
        systemPrompt:
          'You are Quickio AI Ergonomics Specialist. Evaluate neck flexion, lumbar support, elbow angle, and monitor height from the user photo or description, providing RULA-inspired corrective steps.',
      },
    ],
  },
  {
    category: '📚 PRODUCTIVITY & TECH',
    items: [
      {
        id: 'flowchart-generator',
        name: 'Flowchart Generator',
        vision: false,
        isFlowchart: true,
        desc: 'Convert plain text descriptions, algorithms, or workflows into a visual step-by-step flowchart.',
        defaultInstruction:
          'Enter any plain text process, algorithm, or workflow below (e.g., "User login with Google OAuth and 2FA verification" or "Photosynthesis light & dark reactions") to generate a visual flowchart.',
        systemPrompt:
          'You are Quickio AI Flowchart Architect. Transform the user plain-text topic or process into a clear, logical step-by-step flowchart with Start, Process, Decision, and End nodes.',
      },
      {
        id: 'study-planner',
        name: 'Study Planner',
        vision: false,
        isStudyPlanner: true,
        desc: 'Generate spaced-repetition schedules, Pomodoro blocks, and exam mastery roadmaps.',
        defaultInstruction:
          'Default Study Planning Instructions:\n1. Mention your Exam / Subject names and syllabus topics.\n2. State your Target Date or number of days left.\n3. Specify how many hours per day you can study.\n4. Quickio AI will generate a day-by-day Pomodoro + Spaced Repetition timetable.',
        systemPrompt:
          'You are Quickio AI Study Planner. Follow standard instructional design: build a structured day-by-day study schedule with Pomodoro time blocks, active recall checkpoints, and spaced repetition intervals (Day 1, Day 3, Day 7, Day 14).',
      },
      {
        id: 'bilingual-quiz',
        name: 'Bilingual Quiz',
        vision: false,
        isQuiz: true,
        desc: 'Interactive Quickio multiple-choice bilingual quiz with instant grading, hints, and explanations.',
        systemPrompt:
          'You are Quickio AI Bilingual Quiz Engine. Generate interactive multiple-choice questions in two languages with hints and detailed explanations.',
      },
      {
        id: 'spam-fact-audit',
        name: 'Spam & Fact Audit',
        vision: true,
        isSpamAudit: true,
        desc: 'Detect phishing links, scam messages, and fake news, plus official Cyber Crime reporting steps.',
        systemPrompt:
          'You are Quickio AI Spam & Fact Auditor. Analyze the message, screenshot, or claim for phishing indicators, social engineering urgency, spoofed domains, and factual accuracy. Always conclude with clear safety actions and official reporting steps.',
      },
      {
        id: 'crop-weather',
        name: 'Crop & Weather',
        vision: true,
        isCropWeather: true,
        desc: 'Live Network Weather dashboard (Google/MSN Weather style) + AI Agronomy Q&A advisor.',
        systemPrompt:
          'You are Quickio AI Crop & Weather Agronomist. Combine the live local weather telemetry (temperature, humidity, wind, precipitation forecast) with the user answers to provide tailored crop selection, irrigation timing, and pest/weather risk advice.',
      },
    ],
  },
];

export const REQUIREMENTS_TXT = `customtkinter>=5.2.2
Pillow>=10.4.0
requests>=2.32.3
google-genai>=1.5.0
opencv-python>=4.10.0.84
ultralytics>=8.3.0
pyinstaller>=6.11.0
`;

export const BUILD_WINDOWS_EXE_BAT = `@echo off
title Quickio AI — Windows .EXE Builder (CustomTkinter + YOLOv8 + Quickio/Ollama)
color 0D

echo ============================================================================
echo   QUICKIO AI - WINDOWS DESKTOP .EXE COMPILER (PyInstaller)
echo ============================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your Windows PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Installing required packages (CustomTkinter, google-genai, OpenCV, Ultralytics YOLO, PyInstaller)...
python -m pip install --upgrade pip
python -m pip install customtkinter Pillow requests google-genai opencv-python ultralytics pyinstaller

echo.
echo [2/3] Compiling quickio_ai.py into standalone Windows executable (Quickio_AI.exe)...
python -m PyInstaller --noconfirm --onedir --windowed --name "Quickio_AI" --collect-all customtkinter --collect-all ultralytics quickio_ai.py

echo.
echo [3/3] Build Complete!
echo Your standalone Windows executable is ready at:
echo   dist\\Quickio_AI\\Quickio_AI.exe
echo.
if exist "dist\\Quickio_AI\\Quickio_AI.exe" (
    echo Launching Quickio_AI.exe...
    start "" "dist\\Quickio_AI\\Quickio_AI.exe"
)
pause
`;
