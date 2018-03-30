from configobj import ConfigObj


metadata_path = './metadata.cfg'
print('config_path: {}'.format(metadata_path))

metadata = ConfigObj(metadata_path, encoding="UTF8", indent_type='    ', unrepr=True,
                     create_empty=True, write_empty_values=True)
metadata.reload()



