from rich import print

def file_naming():
    
    while True:
        print(" ")
        print("FILE NAMING PROGRAM: Puzzle Data")
        print(" ")

        print("[bold][underline]DATA LEGEND:[/underline][/bold]")
        print("Puzzles: (A), (B), or (C)")
        print("Modalities: audio (U), eye tracking (E), or gamestate (G)")
        print("")


        puzzle = input("In the given file, select which puzzle the data was obtained from:  ")
        modality = input("In the given file, select the type of data extracted:  ")
        PID = input("Please enter the participant's identification number:  ")
        date = input("Please enter the date of data collection: ")

        while puzzle.lower() != "a" and puzzle.lower() != "b" and puzzle.lower() != "c":
            print("Error - invalid entry. Please refer back to data legend.")
            puzzle = input("In the given file, select which puzzle the data was obtained from:  ")

        while True:
            if len(PID) != 3:
                print("Invalid entry - PID must be exactly 3 digits.")
            elif not PID.isdigit():
                print("Invalid entry - PID must contain only numbers.")
            elif int(PID) < 1 or int(PID) > 300:
                print("Invalid entry - number must be between 1 and 300.")
            else:
                break
            PID = input("Please enter the participant's identification number:  ")
        
        while True:
            if not date.isdigit():
                print("Error - invalid entry. Date must be numeric.")
            elif len(date) != 8:
                print("Error - invalid entry. Date must be written in MMDDYYYY format.")
            else:
                break
            date = input("Please enter the date of data collection: ")

        if puzzle.lower() == "a":
            while modality.lower() != "u" and modality.lower() != "e" and modality.lower() != "g":
                print("Error - invalid entry. Please refer back to data legend.")
                modality = input("In the given file, select the type of data extracted:  ")
            if modality.lower() == "u":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gA_audio_{date}.webm")
                print(" ")
            elif modality.lower() == "e":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gA_eyetracking_{date}.json")
                print(" ")
            elif modality.lower() == "g":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gA_gamestate_{date}.json")
                print(" ")

        elif puzzle.lower() == "b":
            while modality.lower() != "u" and modality.lower() != "e" and modality.lower() != "g":
                print("Error - invalid entry. Please refer back to data legend.")
                modality = input("In the given file, select the type of data extracted:  ")
            if modality.lower() == "u":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gB_audio_{date}.webm")
                print(" ")
            elif modality.lower() == "e":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gB_eyetracking_{date}.json")
                print(" ")
            elif modality.lower() == "g":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gB_gamestate_{date}.json")
                print(" ")

        elif puzzle.lower() == "c":
            while modality.lower() != "u" and modality.lower() != "e" and modality.lower() != "g":
                print("Error - invalid entry. Please refer back to data legend.")
                modality = input("In the given file, select the type of data extracted:  ")
            if modality.lower() == "u":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gC_audio_{date}.webm")
                print(" ")
            elif modality.lower() == "e":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gC_eyetracking_{date}.json")
                print(" ")
            elif modality.lower() == "g":
                print(" ")
                print("[bold]COPY FILE NAME:[/bold]")
                print(f"P{PID}_gC_gamestate_{date}.json")
                print(" ")

if __name__ == "__main__":
    file_naming()