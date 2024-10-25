import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

text = 'This is a example of a sentence with grammer mistakes.'
matches = tool.check(text)

for match in matches:
    print(match)
