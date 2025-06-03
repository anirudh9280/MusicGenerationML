#!/usr/bin/env python3

import nbformat as nbf
import re

def convert_py_to_notebook():
    """Convert Task2 Python file to Jupyter notebook with proper markdown handling"""
    
    with open('Task2_Executable_Notebook.py', 'r') as f:
        content = f.read()
    
    nb = nbf.v4.new_notebook()
    cells = re.split(r'^# %%', content, flags=re.MULTILINE)
    
    for cell_content in cells:
        if not cell_content.strip():
            continue
            
        if cell_content.strip().startswith(' [markdown]'):
            # Process markdown cell
            lines = cell_content.split('\n')
            md_lines = []
            
            for line in lines[1:]:  # Skip the [markdown] line
                if line.startswith('# '):
                    md_lines.append(line[2:])  # Remove '# ' prefix
                elif line.startswith('#'):
                    md_lines.append(line[1:])  # Remove '#' prefix  
                else:
                    md_lines.append(line)
            
            md_content = '\n'.join(md_lines).strip()
            if md_content:
                nb.cells.append(nbf.v4.new_markdown_cell(md_content))
        else:
            # Process code cell
            code_content = cell_content.strip()
            if code_content:
                nb.cells.append(nbf.v4.new_code_cell(code_content))
    
    with open('workbook.ipynb', 'w') as f:
        nbf.write(nb, f)
    print('✅ Created workbook.ipynb from Task2_Executable_Notebook.py')

if __name__ == "__main__":
    convert_py_to_notebook() 