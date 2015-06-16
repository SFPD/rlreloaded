from docutils import nodes
from docutils.parsers.rst import Directive, directives
import os.path as osp
import sys,traceback
from StringIO import StringIO
import codecs
NAMESPACE = {}

def eval_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    result = eval(text,NAMESPACE)
    return [nodes.Text(unicode(result))],[]

class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    # http://stackoverflow.com/questions/7250659/python-code-to-generate-part-of-sphinx-documentation-is-it-possible
    has_content = True

    def run(self):
        NAMESPACE["DOC"] = sio = StringIO()
        try:
            exec '\n'.join(self.content) in NAMESPACE
            s = unicode(sio.getvalue().decode('UTF-8'))
            if len(s) > 0:
                return [nodes.literal_block(text = s)]
            else:
                return []
        except Exception, e:
            traceback.print_exc()
            return [nodes.error(None, nodes.paragraph(text = "Unable to execute python code at %s:%d:" % (osp.basename(self.src), self.srcline)), nodes.paragraph(text = str(e)))]

# See subl /Library/Python/2.7/site-packages/sphinx/directives/code.py



class LiteralInclude2(Directive):
    """
    Like ``.. include:: :literal:``, but only warns if the include file is
    not found, and does not raise errors.  Also has several options for
    selecting what to include.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'linenos': directives.flag,
        'tab-width': int,
        'language': directives.unchanged_required,
        'encoding': directives.encoding,
        'pyobject': directives.unchanged_required,
        'lines': directives.unchanged_required,
        'start-before': directives.unchanged_required,
        'end-after': directives.unchanged_required,
        'prepend': directives.unchanged_required,
        'append': directives.unchanged_required,
        'emphasize-lines': directives.unchanged_required,
    }

    def run(self):
        document = self.state.document
        if not document.settings.file_insertion_enabled:
            return [document.reporter.warning('File insertion disabled',
                                              line=self.lineno)]
        filename = osp.expandvars(self.arguments[0])
        env = document.settings.env

        if 'pyobject' in self.options and 'lines' in self.options:
            return [document.reporter.warning(
                'Cannot use both "pyobject" and "lines" options',
                line=self.lineno)]

        encoding = self.options.get('encoding', env.config.source_encoding)
        codec_info = codecs.lookup(encoding)
        f = None
        try:
            f = codecs.StreamReaderWriter(open(filename, 'rb'),
                    codec_info[2], codec_info[3], 'strict')
            lines = f.readlines()
        except (IOError, OSError):
            return [document.reporter.warning(
                'Include file %r not found or reading it failed' % filename,
                line=self.lineno)]
        except UnicodeError:
            return [document.reporter.warning(
                'Encoding %r used for reading included file %r seems to '
                'be wrong, try giving an :encoding: option' %
                (encoding, filename))]
        finally:
            if f is not None:
                f.close()

        objectname = self.options.get('pyobject')
        if objectname is not None:
            from sphinx.pycode import ModuleAnalyzer
            analyzer = ModuleAnalyzer.for_file(filename, '')
            tags = analyzer.find_tags()
            if objectname not in tags:
                return [document.reporter.warning(
                    'Object named %r not found in include file %r' %
                    (objectname, filename), line=self.lineno)]
            else:
                lines = lines[tags[objectname][1]-1 : tags[objectname][2]-1]

        linespec = self.options.get('lines')
        if linespec is not None:
            try:
                linelist = parselinenos(linespec, len(lines))
            except ValueError, err:
                return [document.reporter.warning(str(err), line=self.lineno)]
            # just ignore nonexisting lines
            nlines = len(lines)
            lines = [lines[i] for i in linelist if i < nlines]
            if not lines:
                return [document.reporter.warning(
                    'Line spec %r: no lines pulled from include file %r' %
                    (linespec, filename), line=self.lineno)]

        linespec = self.options.get('emphasize-lines')
        if linespec:
            try:
                hl_lines = [x+1 for x in parselinenos(linespec, len(lines))]
            except ValueError, err:
                return [document.reporter.warning(str(err), line=self.lineno)]
        else:
            hl_lines = None

        startafter = self.options.get('start-before')
        endbefore  = self.options.get('end-after')
        prepend    = self.options.get('prepend')
        append     = self.options.get('append')
        if startafter is not None or endbefore is not None:
            use = not startafter
            res = []
            for line in lines:
                if not use and startafter and startafter in line:
                    use = True
                elif use and endbefore and endbefore in line:
                    use = False
                    res.append(line)
                    break
                if use:
                    res.append(line)
            lines = res

        if prepend:
            lines.insert(0, prepend + '\n')
        if append:
            lines.append(append + '\n')

        text = ''.join(lines)
        if self.options.get('tab-width'):
            text = text.expandtabs(self.options['tab-width'])
        retnode = nodes.literal_block(text, text, source=filename)
        # set_source_info(self, retnode)
        if self.options.get('language', ''):
            retnode['language'] = self.options['language']
        if 'linenos' in self.options:
            retnode['linenos'] = True
        if hl_lines is not None:
            retnode['highlight_args'] = {'hl_lines': hl_lines}
        # env.note_dependency(rel_filename)
        return [retnode]



def setup(app):
    """Install the plugin.

    :param app: Sphinx application context.
    """
    app.add_role('eval', eval_role)
    app.add_directive('exec',ExecDirective)
    app.add_directive('literalinclude2',LiteralInclude2)
