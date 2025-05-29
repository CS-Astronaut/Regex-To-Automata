# Regular Expression to DFA Converter

This project implements a tool that converts regular expressions to Finite Automata (DFA & NFA) with visualization capabilities. It provides a step-by-step conversion process from regular expressions to NFA (Non-deterministic Finite Automaton) and then to DFA, including minimization of the resulting DFA.

## Features

- Regular expression parsing and AST construction
- Thompson's construction algorithm for converting regex to NFA
- Subset construction algorithm for converting NFA to DFA
- Hopcroft's algorithm for DFA minimization
- Visualization of both NFA and DFA using Graphviz
- Support for basic regular expression operators:
  - Concatenation (implicit)
  - Union (`U`)
  - Kleene star (`*`)
  - Parentheses for grouping
  - Epsilon transitions (`ε`)
  - Binary alphabet (`0` and `1`)

---

## Requirements

- Python 3.x
- Graphviz (for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CS-Astronaut/Regex-To-Automata
cd Regex-To-Automata
```

2. Install the required Python package:
```bash
pip install graphviz
```
---

## Usage

Run the program:
```bash
python main.py
```

When prompted, enter a regular expression using the following syntax:
- Use `U` for union (e.g., `0U1` for "0 or 1")
- Use `*` for Kleene star (e.g., `0*` for "zero or more 0s")
- Use `ε` for epsilon transitions
- Use parentheses for grouping (e.g., `(0U1)*`)

The program will generate two visualization files:
- `output_nfa.png`: Visual representation of the NFA
- `output_dfa.png`: Visual representation of the minimized DFA

## Example

Input regular expression: `(0U1)*`

This will generate:
1. An NFA using Thompson's construction
2. A DFA using subset construction
3. A minimized DFA using Hopcroft's algorithm
4. Visual representations of both the NFA and DFA

---

## Implementation Details

The project consists of several key components:

1. **Parser**: Converts regular expressions into an Abstract Syntax Tree (AST)
2. **Thompson Construction**: Converts AST to NFA
3. **Subset Construction**: Converts NFA to DFA
4. **Hopcroft Minimization**: Minimizes the DFA
5. **Visualization**: Generates visual representations using Graphviz


---

## ScreenShots

### Ex1
`(0U1)*(0(0U1)*0)(0U1)*` = Representing A Language That Contains The Strings With **At least two 0s**



![DFA for (0U1)*(0(0U1)*0)(0U1)*](/examples/(0U1)*(0(0U1)*0)(0U1)*/output_dfa.png)
![NFA for (0U1)*(0(0U1)*0)(0U1)*](/examples/(0U1)*(0(0U1)*0)(0U1)*/output_nfa.png)


### Ex2
`(0U1)*1011(0U1)*` = Representing A Language That Contains The Strings With **Substring of 1011**

![DFA for (0U1)*1011(0U1)*](/examples/(0U1)*1011(0U1)*/output_dfa.png)
![NFA for (0U1)*1011(0U1)*](/examples/(0U1)*1011(0U1)*/output_nfa.png)

### Ex3
`1*01*01*` = Representing A Language That Contains The Strings With **Exactly two 0s**

![DFA for 1*01*01*](/examples/1*01*01*/output_dfa.png)
![NFA for 1*01*01*](/examples/1*01*01*/output_nfa.png)

### Ex3 Union Ex2 !

*Also Can Draw The Machine That Accepts The Union Of Two Languages*
 
`(1*01*01*)U((0U1)*1011(0U1)*)` = Representing A Language That Contains The Strings With **Substring of 1011** or **Exactly two 0s**


![DFA for (1*01*01*)U((0U1)*1011(0U1)*)](/examples/(1*01*01*)U((0U1)*1011(0U1)*)/output_dfa.png)
![NFA for (1*01*01*)U((0U1)*1011(0U1)*)](/examples/(1*01*01*)U((0U1)*1011(0U1)*)/output_nfa.png)


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
