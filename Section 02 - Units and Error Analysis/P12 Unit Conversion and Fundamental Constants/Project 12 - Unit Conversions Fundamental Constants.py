#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np

def unit_conversion():
    print("Welcome to the Advanced Unit Conversion Tool!")
    print("Choose the type of conversion you want to perform:")
    print("1. Length")
    print("2. Mass")
    print("3. Time")
    print("4. Volume")
    print("5. Energy")
    print("6. Temperature")
    print("7. Amount (Moles/Particles)")
    print("8. Pressure")
    print("9. Velocity")
    choice = input("Enter your choice (1-9): ")

    if choice == "1":  # Length conversions
        print("Length conversion:")
        print("Choose the units to convert from:")
        print("1. Meters (m)")
        print("2. Kilometers (km)")
        print("3. Miles (mi)")
        print("4. Inches (in)")
        print("5. Feet (ft)")
        print("6. Astronomical Units (AU)")
        unit_from = input("Enter your choice: ")
        value = float(input("Enter the value: "))
        print("Choose the units to convert to:")
        unit_to = input("Enter your choice: ")

        conversion_factors = {
            ("1", "2"): 0.001,  # meters to kilometers
            ("1", "3"): 0.000621371,  # meters to miles
            ("1", "4"): 39.3701,  # meters to inches
            ("1", "5"): 3.28084,  # meters to feet
            ("1", "6"): 6.684587e-12,  # meters to astronomical units
            ("2", "3"): 0.621371,  # kilometers to miles
            ("2", "4"): 39370.1,  # kilometers to inches
            ("2", "5"): 3280.84,  # kilometers to feet
            ("2", "6"): 6.684587e-9,  # kilometers to astronomical units
            ("3", "4"): 63360,  # miles to inches
            ("3", "5"): 5280,  # miles to feet
            ("3", "6"): 1.07578e-8,  # miles to astronomical units
            ("4", "5"): 0.0833333,  # inches to feet
            ("4", "6"): 1.697885e-13,  # inches to astronomical units
            ("5", "6"): 2.03125e-14,  # feet to astronomical units
        }

        # Ensure all permutations are supported
        if (unit_from, unit_to) in conversion_factors:
            factor = conversion_factors[(unit_from, unit_to)]
            result = value * factor
        elif (unit_to, unit_from) in conversion_factors:
            factor = conversion_factors[(unit_to, unit_from)]
            result = value / factor
        else:
            print("Conversion not supported!")
            return
        print(f"Converted value: {result}")

    elif choice == "2":  # Mass conversions
        print("Mass conversion:")
        print("Choose the units to convert from:")
        print("1. Kilograms (kg)")
        print("2. Grams (g)")
        print("3. Pounds (lbs)")
        print("4. Ounces (oz)")
        print("5. Atomic Mass Units (amu)")
        unit_from = input("Enter your choice: ")
        value = float(input("Enter the value: "))
        print("Choose the units to convert to:")
        unit_to = input("Enter your choice: ")

        conversion_factors = {
            ("1", "2"): 1000,  # kilograms to grams
            ("1", "3"): 2.20462,  # kilograms to pounds
            ("1", "4"): 35.274,  # kilograms to ounces
            ("1", "5"): 6.022e26,  # kilograms to amu
            ("2", "3"): 0.00220462,  # grams to pounds
            ("2", "4"): 0.035274,  # grams to ounces
            ("2", "5"): 6.022e23,  # grams to amu
            ("3", "4"): 16,  # pounds to ounces
            ("3", "5"): 2.7316e25,  # pounds to amu
            ("4", "5"): 1.7072e24,  # ounces to amu
        }

        if (unit_from, unit_to) in conversion_factors:
            factor = conversion_factors[(unit_from, unit_to)]
            result = value * factor
        elif (unit_to, unit_from) in conversion_factors:
            factor = conversion_factors[(unit_to, unit_from)]
            result = value / factor
        else:
            print("Conversion not supported!")
            return
        print(f"Converted value: {result}")

    elif choice == "3":  # Time conversions
        print("Time conversion:")
        print("Choose the units to convert from:")
        print("1. Microseconds (µs)")
        print("2. Milliseconds (ms)")
        print("3. Seconds (s)")
        print("4. Minutes (min)")
        print("5. Hours (hr)")
        print("6. Days (day)")
        print("7. Years (yr)")
        unit_from = input("Enter your choice: ")
        value = float(input("Enter the value: "))
        print("Choose the units to convert to:")
        unit_to = input("Enter your choice: ")

        conversion_factors = {
            ("1", "2"): 0.001,  # microseconds to milliseconds
            ("1", "3"): 1e-6,  # microseconds to seconds
            ("2", "3"): 0.001,  # milliseconds to seconds
            ("3", "4"): 1/60,  # seconds to minutes
            ("3", "5"): 1/3600,  # seconds to hours
            ("3", "6"): 1/86400,  # seconds to days
            ("3", "7"): 1/3.154e7,  # seconds to years
            ("4", "5"): 1/60,  # minutes to hours
            ("4", "6"): 1/1440,  # minutes to days
            ("4", "7"): 1/525600,  # minutes to years
            ("5", "6"): 1/24,  # hours to days
            ("5", "7"): 1/8760,  # hours to years
            ("6", "7"): 1/365.25,  # days to years
        }

        if (unit_from, unit_to) in conversion_factors:
            factor = conversion_factors[(unit_from, unit_to)]
            result = value * factor
        elif (unit_to, unit_from) in conversion_factors:
            factor = conversion_factors[(unit_to, unit_from)]
            result = value / factor
        else:
            print("Conversion not supported!")
            return
        print(f"Converted value: {result}")


    elif choice == "5":  # Energy conversions
            print("Energy conversion:")
            print("Choose the units to convert from:")
            print("1. Joules (J)")
            print("2. Millijoules (mJ)")
            print("3. Kilojoules (kJ)")
            print("4. Gigajoules (GJ)")
            print("5. Electronvolts (eV)")
            print("6. Mega-electronvolts (MeV)")
            print("7. Giga-electronvolts (GeV)")
            print("8. Kilowatt-hours (kWh)")
            print("9. Calories (cal)")
            print("10. Kilocalories (kCal)")
            unit_from = input("Enter your choice: ")
            value = float(input("Enter the value: "))
            print("Choose the units to convert to:")
            unit_to = input("Enter your choice: ")
    
            conversion_factors = {
                ("1", "2"): 1000,  # J to mJ
                ("1", "3"): 0.001,  # J to kJ
                ("1", "4"): 1e-9,  # J to GJ
                ("1", "5"): 6.242e18,  # J to eV
                ("1", "6"): 6.242e12,  # J to MeV
                ("1", "7"): 6.242e9,  # J to GeV
                ("1", "8"): 2.7778e-7,  # J to kWh
                ("1", "9"): 0.239006,  # J to cal
                ("1", "10"): 0.000239006,  # J to kcal
                ("9", "10"): 0.001,  # cal to kcal
            }
    
            if (unit_from, unit_to) in conversion_factors:
                factor = conversion_factors[(unit_from, unit_to)]
                result = value * factor
            elif (unit_to, unit_from) in conversion_factors:
                factor = conversion_factors[(unit_to, unit_from)]
                result = value / factor
            else:
                print("Conversion not supported!")
                return
            print(f"Converted value: {result}")

    elif choice == "6":  # Temperature conversions
        print("Temperature conversion:")
        print("Choose the units to convert from:")
        print("1. Celsius (°C)")
        print("2. Fahrenheit (°F)")
        print("3. Kelvin (K)")
        unit_from = input("Enter your choice: ")
        value = float(input("Enter the value: "))
        print("Choose the units to convert to:")
        unit_to = input("Enter your choice: ")

        if unit_from == "1" and unit_to == "2":  # °C to °F
            result = (value * 9/5) + 32
        elif unit_from == "1" and unit_to == "3":  # °C to K
            result = value + 273.15
        elif unit_from == "2" and unit_to == "1":  # °F to °C
            result = (value - 32) * 5/9
        elif unit_from == "2" and unit_to == "3":  # °F to K
            result = ((value - 32) * 5/9) + 273.15
        elif unit_from == "3" and unit_to == "1":  # K to °C
            result = value - 273.15
        elif unit_from == "3" and unit_to == "2":  # K to °F
            result = ((value - 273.15) * 9/5) + 32
        else:
            print("Conversion not supported!")
            return
        print(f"Converted value: {result}")

    elif choice == "7":  # Amount conversions
        print("Amount conversion:")
        print("Choose the units to convert from:")
        print("1. Moles (mol)")
        print("2. Particles")
        unit_from = input("Enter your choice: ")
        value = float(input("Enter the value: "))
        print("Choose the units to convert to:")
        unit_to = input("Enter your choice: ")

        avogadro_number = 6.022e23  # Particles per mole
        if unit_from == "1" and unit_to == "2":  # Moles to particles
            result = value * avogadro_number
        elif unit_from == "2" and unit_to == "1":  # Particles to moles
            result = value / avogadro_number
        else:
            print("Conversion not supported!")
            return
        print(f"Converted value: {result}")

    elif choice == "8":  # Pressure conversions
        print("Pressure conversion:")
        print("Choose the units to convert from:")
        print("1. Atmospheres (atm)")
        print("2. Pascals (Pa)")
        print("3. Bar (bar)")
        print("4. Millibar (mbar)")
        print("5. Torr (mmHg)")
        print("6. Pounds per square inch (psi)")
        unit_from = input("Enter your choice: ")
        value = float(input("Enter the value: "))
        print("Choose the units to convert to:")
        unit_to = input("Enter your choice: ")

        conversion_factors = {
            ("1", "2"): 101325,  # atm to Pa
            ("1", "3"): 1.01325,  # atm to bar
            ("1", "4"): 1013.25,  # atm to mbar
            ("1", "5"): 760,  # atm to Torr
            ("1", "6"): 14.696,  # atm to psi
            ("2", "3"): 1e-5,  # Pa to bar
            ("2", "4"): 0.01,  # Pa to mbar
            ("2", "5"): 7.50062e-3,  # Pa to Torr
            ("2", "6"): 1.45038e-4,  # Pa to psi
            ("3", "4"): 1000,  # bar to mbar
            ("3", "5"): 750.062,  # bar to Torr
            ("3", "6"): 14.5038,  # bar to psi
            ("4", "5"): 0.750062,  # mbar to Torr
            ("4", "6"): 0.0145038,  # mbar to psi
            ("5", "6"): 0.0193368,  # Torr to psi
        }

        # Handle bidirectional conversions
        if (unit_from, unit_to) in conversion_factors:
            factor = conversion_factors[(unit_from, unit_to)]
            result = value * factor
        elif (unit_to, unit_from) in conversion_factors:
            factor = conversion_factors[(unit_to, unit_from)]
            result = value / factor
        else:
            print("Conversion not supported!")
            return
        print(f"Converted value: {result}")
    elif choice == "9":  # Velocity conversions
        print("Velocity conversion:")
        print("Choose the units to convert from:")
        print("1. Meters per second (m/s)")
        print("2. Kilometers per second (km/s)")
        print("3. Kilometers per hour (km/h)")
        print("4. Miles per hour (mph)")
        print("5. Fraction of the speed of light (c)")
        unit_from = input("Enter your choice: ")
        value = float(input("Enter the value: "))
        print("Choose the units to convert to:")
        unit_to = input("Enter your choice: ")

        speed_of_light = 3e8  # Speed of light in m/s
        conversion_factors = {
            ("1", "2"): 0.001,  # m/s to km/s
            ("1", "3"): 3.6,  # m/s to km/h
            ("1", "4"): 2.23694,  # m/s to mph
            ("1", "5"): 1 / speed_of_light,  # m/s to c
            ("3", "4"): 0.621371,  # km/h to mph
        }

        if (unit_from, unit_to) in conversion_factors:
            factor = conversion_factors[(unit_from, unit_to)]
            result = value * factor
        elif (unit_to, unit_from) in conversion_factors:
            factor = conversion_factors[(unit_to, unit_from)]
            result = value / factor
        else:
            print("Conversion not supported!")
            return
        print(f"Converted value: {result}")

    else:
        print("Invalid choice or not yet implemented! Please enter a valid number.")
while True:
    unit_conversion()  # Call the unit conversion function
    choice = input("\nWould you like to perform another conversion? (yes/no): ").strip().lower()
    if choice != "yes":
        print("Exiting the Unit Conversion Tool.)
        break


# In[23]:


import pandas as pd

# Constructing the table of fundamental constants again in an easily displayable format
constants_data = {
    "Constant": [
        "Pi (π)", "Euler's Number (e)", "Speed of Light (c)", "Planck Constant (h)",
        "Planck Constant (h) in eV·s", "Planck Length", "Planck Time",
        "Gravitational Constant (G)", "Molar Gas Constant (R)",
        "Avogadro's Number", "Electron Charge (e)", "Permittivity of Free Space (ε₀)",
        "Permeability of Free Space (μ₀)", "Mass of Electron (mₑ)",
        "Mass of Proton (mₚ)", "Mass of Neutron (mₙ)", "Boltzmann Constant (k₆)",
        "Stefan-Boltzmann Constant (σ)", "Rydberg Constant (R∞)",
        "Bohr Radius (a₀)", "Specific Heat of Water",
        "Speed of Sound (at room temperature)", "Hubble Constant",
        "Age of Universe (years)"
    ],
    "Value": [
        "3.14159265359", "2.71828182846", "299,792,458", "6.62607015e-34",
        "4.135667696e-15", "1.616255e-35", "5.391247e-44",
        "6.67430e-11", "8.314462618",
        "6.02214076e23", "1.602176634e-19", "8.854187817e-12",
        "1.25663706212e-6", "9.10938356e-31",
        "1.67262192369e-27", "1.6754924e-27", "1.380649e-23",
        "5.670374419e-8", "10,973,731.56816",
        "5.29177210903e-11", "4,186",
        "343", "67.4 (km/s/Mpc)",
        "13.8 billion years"
    ],
    "Units": [
        "Dimensionless", "Dimensionless", "m/s", "J·s",
        "eV·s", "m", "s",
        "m³·kg⁻¹·s⁻²", "J·mol⁻¹·K⁻¹",
        "mol⁻¹", "C", "F·m⁻¹",
        "N·A⁻²", "kg",
        "kg", "kg", "J·K⁻¹",
        "W·m⁻²·K⁻⁴", "m⁻¹",
        "m", "J·kg⁻¹·K⁻¹",
        "m/s", "km/s/Mpc",
        "Years"
    ]
}

# Create a DataFrame for better display
constants_df = pd.DataFrame(constants_data)

# Outputting the table in a text-friendly format for the user's environment
print(constants_df)


# In[ ]:




