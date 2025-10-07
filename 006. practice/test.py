import textwrap

text = "ABCDEFGHIJKLIMNOQRSTUVWXYZ"

# Basic wrap into list of lines
lines = textwrap.wrap(text, width=4)
for line in lines:
    print(line)

# # Fill into a single string with newlines
# filled = textwrap.fill(text, width=30)
# print(filled)

# # Dedent a triple-quoted string
# s = """
#     This is indented
#       more here
#     less here
# """
# print(textwrap.dedent(s))

# # Shorten with placeholder
# short = textwrap.shorten("This is an example of a long text we need to truncate", width=20)
# print(short)  # e.g. "This is an [...]"

# # Indent lines with prefix
# sample = "Line one\nLine two\nAnother line"
# indented = textwrap.indent(sample, prefix="> ")
# print(indented)