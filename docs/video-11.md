Text splitters

-> To opt

Types

1. Length based text splitting

- Split in chunks with specific size(characters or tokens)
- Tool for different config splitting: https://chunkviz.up.railway.app/
- Input
    - chunk size
    - Chunk overlap size (According to lanchain 10-20 percent of chunk size)

- Advantage
    - Fast.
- Disadvantage
    - Do not see contextual meaning, semantic meaning while splitting. SO word or sentence or paragraph may be cut in
      between

2. Text-structure Based splitting
-  Splitters: '\n\n' (paragraph), '\n' (sentences), ' ' (space), '.' (character)
- Input
  - Chunk size
- Create chunk based on chunk size and split based on splitter.
- Tries to not break character in between.




3. Document structure based
-