#!/usr/bin/env python3
"""
DriftDetector v2 - Interactive Setup Wizard
Guides users through installation options and configuration
"""

import sys
import os
import shutil
from pathlib import Path

# ============================================================================
# ANSI Colors for terminal output
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ============================================================================
# Setup Presets
# ============================================================================

PRESETS = {
    "1": {
        "name": "Core Only",
        "description": "Local model inference (OpenClaw, Ollama) - no API keys needed",
        "install": "pip install drift-detector",
        "features": ["Core Drift Detection"],
        "apis": []
    },
    "2": {
        "name": "Core + UI",
        "description": "Add monitoring dashboard - real-time drift tracking & trends",
        "install": "pip install drift-detector[ui]",
        "features": ["Core Drift Detection", "Web Dashboard"],
        "apis": ["OPENCLAW_HOST", "OLLAMA_HOST"]
    },
    "3": {
        "name": "Core + LangChain",
        "description": "Integrate with LangChain agents - auto drift detection",
        "install": "pip install drift-detector[langchain]",
        "features": ["Core Drift Detection", "LangChain Integration"],
        "apis": ["GROQ_API_KEY", "CEREBRAS_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    },
    "4": {
        "name": "Core + CrewAI",
        "description": "Multi-agent teams - agent coordination with drift tracking",
        "install": "pip install drift-detector[crewai]",
        "features": ["Core Drift Detection", "CrewAI Integration"],
        "apis": ["GROQ_API_KEY", "CEREBRAS_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    },
    "5": {
        "name": "Full Stack",
        "description": "Everything - monitoring dashboard + all integrations",
        "install": "pip install drift-detector[all]",
        "features": ["Core Drift Detection", "Web Dashboard", "LangChain", "CrewAI"],
        "apis": ["OPENCLAW_HOST", "OLLAMA_HOST", "GROQ_API_KEY", "CEREBRAS_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    },
    "6": {
        "name": "Custom",
        "description": "Build your own - pick exactly what you need",
        "install": None,  # Will be determined by user choices
        "features": [],
        "apis": []
    }
}

# ============================================================================
# Main Wizard
# ============================================================================

def print_header():
    """Print welcome header"""
    print()
    print(f"{Colors.HEADER}{Colors.BOLD}═" * 70 + f"{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}   DriftDetector v2 - Setup Wizard{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}═" * 70 + f"{Colors.RESET}")
    print()
    print("Welcome! This wizard will guide you through DriftDetector installation.")
    print("Choose your setup and we'll generate the perfect configuration for you.")
    print()

def show_preset_menu():
    """Show preset options"""
    print(f"{Colors.CYAN}Choose your setup:{Colors.RESET}")
    print()

    for key, preset in PRESETS.items():
        features_str = ", ".join(preset["features"]) if preset["features"] else "Custom"
        print(f"{Colors.BOLD}{key}{Colors.RESET}) {Colors.GREEN}{preset['name']}{Colors.RESET}")
        print(f"   {preset['description']}")
        print(f"   Features: {features_str}")
        print()

def get_preset_choice():
    """Get user's preset choice"""
    while True:
        choice = input(f"{Colors.BLUE}Select option (1-6): {Colors.RESET}").strip()
        if choice in PRESETS:
            return choice
        print(f"{Colors.RED}Invalid choice. Please select 1-6.{Colors.RESET}")

def get_custom_choices():
    """For custom preset: ask what user wants"""
    print()
    print(f"{Colors.CYAN}Build your custom setup:{Colors.RESET}")
    print()

    features = {}

    # UI
    ui = input(f"Add UI Dashboard (Web monitoring)? (y/n): {Colors.RESET}").strip().lower() == 'y'
    features['ui'] = ui

    # LangChain
    langchain = input(f"Add LangChain integration? (y/n): {Colors.RESET}").strip().lower() == 'y'
    features['langchain'] = langchain

    # CrewAI
    crewai = input(f"Add CrewAI integration? (y/n): {Colors.RESET}").strip().lower() == 'y'
    features['crewai'] = crewai

    return features

def generate_install_command(choice, custom_features=None):
    """Generate pip install command"""
    if choice in ["1", "2", "3", "4", "5"]:
        return PRESETS[choice]["install"]

    # Custom
    extras = []
    if custom_features.get('ui'):
        extras.append("ui")
    if custom_features.get('langchain'):
        extras.append("langchain")
    if custom_features.get('crewai'):
        extras.append("crewai")

    if extras:
        return f"pip install drift-detector[{','.join(extras)}]"
    else:
        return "pip install drift-detector"

def get_required_apis(choice, custom_features=None):
    """Get list of APIs needed for this setup"""
    if choice in PRESETS and choice != "6":
        return PRESETS[choice]["apis"]

    # Custom
    apis = []
    if custom_features.get('ui'):
        apis.extend(["OPENCLAW_HOST", "OLLAMA_HOST"])
    if custom_features.get('langchain') or custom_features.get('crewai'):
        apis.extend(["GROQ_API_KEY", "CEREBRAS_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"])

    return list(set(apis))  # Remove duplicates

def copy_env_template():
    """Copy .env.example to .env if it doesn't exist"""
    env_example = Path(".env.example")
    env_file = Path(".env")

    if not env_example.exists():
        print(f"{Colors.YELLOW}⚠ .env.example not found. Skipping .env generation.{Colors.RESET}")
        return False

    if env_file.exists():
        overwrite = input(f".env already exists. Overwrite? (y/n): {Colors.RESET}").strip().lower() == 'y'
        if not overwrite:
            print(f"{Colors.YELLOW}Keeping existing .env{Colors.RESET}")
            return True

    try:
        shutil.copy(".env.example", ".env")
        print(f"{Colors.GREEN}✓ Created .env from template{Colors.RESET}")
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to create .env: {e}{Colors.RESET}")
        return False

def health_check(choice=None):
    """Verify installation"""
    print()
    print(f"{Colors.CYAN}Running health check...{Colors.RESET}")
    print()

    try:
        from drift_detector.core import DriftDetectorAgent
        print(f"{Colors.GREEN}✓ DriftDetector Core imported successfully{Colors.RESET}")

        # Try to instantiate
        from drift_detector.core.drift_detector_agent import AgentConfig
        config = AgentConfig(agent_id="health_check")
        detector = DriftDetectorAgent(config)
        print(f"{Colors.GREEN}✓ DriftDetector agent initialized successfully{Colors.RESET}")

        return True
    except ImportError as e:
        print(f"{Colors.RED}✗ Import failed: {e}{Colors.RESET}")
        if choice and choice in PRESETS:
            print(f"   Make sure to run: {PRESETS[choice]['install']}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ Health check failed: {e}{Colors.RESET}")
        return False

def show_summary(choice, custom_features=None):
    """Show configuration summary"""
    print()
    print(f"{Colors.HEADER}{Colors.BOLD}Setup Summary{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}═" * 70 + f"{Colors.RESET}")
    print()

    if choice in PRESETS and choice != "6":
        preset = PRESETS[choice]
        print(f"Preset: {Colors.GREEN}{preset['name']}{Colors.RESET}")
        print(f"Description: {preset['description']}")
    else:
        print(f"Preset: {Colors.GREEN}Custom{Colors.RESET}")
        features = []
        if custom_features.get('ui'):
            features.append("UI Dashboard")
        if custom_features.get('langchain'):
            features.append("LangChain")
        if custom_features.get('crewai'):
            features.append("CrewAI")
        print(f"Features: {', '.join(features) if features else 'Core only'}")

    print()
    install_cmd = generate_install_command(choice, custom_features)
    print(f"Installation command:")
    print(f"  {Colors.BOLD}{install_cmd}{Colors.RESET}")
    print()

    apis = get_required_apis(choice, custom_features)
    if apis:
        print(f"Required API keys in .env:")
        for api in sorted(apis):
            print(f"  - {api}")
    else:
        print("No API keys required (local models only)")
    print()

def show_next_steps(choice):
    """Show next steps"""
    print(f"{Colors.CYAN}Next steps:{Colors.RESET}")
    print()
    print(f"1. Edit .env with your API keys (if needed)")
    print()

    if choice == "2":
        print(f"2. Start the UI dashboard:")
        print(f"   python -m drift_detector.ui.server")
        print()

    print(f"3. Check examples:")
    print(f"   ls examples/")
    print()

    print(f"4. Read documentation:")
    print(f"   cat README.md")
    print()

def main():
    """Main wizard flow"""
    print_header()

    show_preset_menu()

    choice = get_preset_choice()
    custom_features = None

    if choice == "6":
        custom_features = get_custom_choices()

    show_summary(choice, custom_features)

    # Ask for confirmation
    print()
    confirm = input(f"{Colors.BLUE}Proceed with this setup? (y/n): {Colors.RESET}").strip().lower() == 'y'

    if not confirm:
        print(f"{Colors.YELLOW}Setup cancelled.{Colors.RESET}")
        return

    # Generate .env
    copy_env_template()

    # Health check
    health_check(choice)

    print()
    show_next_steps(choice)

    print(f"{Colors.GREEN}{Colors.BOLD}✓ Setup complete!{Colors.RESET}")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup cancelled.{Colors.RESET}")
        sys.exit(0)
