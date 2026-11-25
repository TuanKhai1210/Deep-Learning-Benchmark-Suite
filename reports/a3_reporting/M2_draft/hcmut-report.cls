\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmut-report}[2026/04/21 HCMUT report class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[a4paper,oneside]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[english]{babel}
\RequirePackage{anyfontsize}
\RequirePackage{amsmath,amssymb}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\RequirePackage{booktabs,longtable,array}
\RequirePackage[center]{caption}
\RequirePackage{fancyhdr}
\RequirePackage{hyperref}
\RequirePackage[super,square,sort]{natbib}
\RequirePackage{chngcntr}
\RequirePackage{titlesec}
\RequirePackage{xcolor}
\usepackage{float}
\usepackage{listings}

\parindent 0ex
\geometry{
  a4paper,
  left=18mm,
  right=18mm,
  top=20mm,
  bottom=25mm,
  includehead,
  includefoot,
  headsep=8mm,
  footskip=12mm
}
\setstretch{1.2}
\raggedbottom
\setlength{\parindent}{1.25cm}
\renewcommand{\arraystretch}{1.2}

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

\makeatletter

\renewcommand\normalsize{%
  \@setfontsize\normalsize{13pt}{15.6pt}%
  \abovedisplayskip 12pt plus 3pt minus 7pt%
  \abovedisplayshortskip \z@ plus 3pt%
  \belowdisplayshortskip 7pt plus 3pt minus 4pt%
  \belowdisplayskip \abovedisplayskip%
  \let\@listi\@listI
}
\normalsize

\titleformat{\section}
  {\bfseries\fontsize{15}{18}\selectfont}
  {\thesection}{1em}{}
\titleformat{\subsection}
  {\bfseries\fontsize{14}{16}\selectfont}
  {\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\bfseries\fontsize{13}{15}\selectfont}
  {\thesubsubsection}{1em}{}

\titlespacing*{\section}{0pt}{12pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{5pt}
\titlespacing*{\subsubsection}{0pt}{8pt}{4pt}

\counterwithin{equation}{section}
\counterwithin{table}{section}
\counterwithin{figure}{section}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=blue
}

\renewcommand{\refname}{References}
\AtBeginDocument{\renewcommand{\contentsname}{Table of Contents}}
\AtBeginDocument{\renewcommand{\tablename}{Table}}
\AtBeginDocument{\renewcommand{\figurename}{Figure}}

\newcommand{\Month}{%
  \ifcase\the\month
  \or January
  \or February
  \or March
  \or April
  \or May
  \or June
  \or July
  \or August
  \or September
  \or October
  \or November
  \or December
  \fi
}

\lstdefinestyle{pythoncode}{
  language=Python,
  basicstyle=\ttfamily\normalsize,
  keywordstyle=\color{blue},
  commentstyle=\itshape\color{gray},
  stringstyle=\color{olive},
  columns=fullflexible,
  frame=single,
  breaklines=true,
  showstringspaces=false,
  numbersep=5pt,
  backgroundcolor=\color{white},
  numbers=none,
}

\newcommand{\@upperuniname}{Vietnam National University Ho Chi Minh City}
\newcommand{\upperuniname}[1]{\gdef\@upperuniname{#1}}

\newcommand{\@uniname}{Ho Chi Minh City University of Technology}
\newcommand{\uniname}[1]{\gdef\@uniname{#1}}

\newcommand{\@deptname}{Faculty of Applied Science}
\newcommand{\deptname}[1]{\gdef\@deptname{#1}}

\newcommand{\@coursename}{MT2013 Probability and Statistics}
\newcommand{\coursename}[1]{\gdef\@coursename{#1}}

\newcommand{\@reporttype}{Project Report}
\newcommand{\reporttype}[1]{\gdef\@reporttype{#1}}

\renewcommand{\title}[1]{\gdef\@title{#1}}
\title{Report title}

\newcommand{\@advisor}{Advisor}
\newcommand{\advisor}[1]{\gdef\@advisor{#1}}

\newcommand{\@groupname}{Group 01}
\newcommand{\groupname}[1]{\gdef\@groupname{#1}}

\newcommand{\@reportplace}{Ho Chi Minh City}
\newcommand{\reportplace}[1]{\gdef\@reportplace{#1}}

\newcommand{\@reporttime}{\Month\ \the\year}
\newcommand{\reporttime}[1]{\gdef\@reporttime{#1}}

\newcommand{\@logofile}{hcmut.png}
\newcommand{\logofile}[1]{\gdef\@logofile{#1}}

\newcommand{\@membertable}{1 & [Name] & [Student ID] & [Major] & [Role] & 100\\\hline}
\newcommand{\membertable}[1]{\gdef\@membertable{#1}}

\setlength{\headheight}{44pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{%
  \begin{tabular}{@{}l@{}}
    \textbf{\@uniname} \\
    \textbf{\@deptname}
  \end{tabular}
}
\fancyhead[R]{%
  \IfFileExists{\@logofile}{\includegraphics[width=10mm]{\@logofile}}{}
}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}

\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyhead[L]{%
    \begin{tabular}{@{}l@{}}
      \textbf{\@uniname} \\\\
      \textbf{\@deptname}
    \end{tabular}
  }
  \fancyhead[R]{%
    \IfFileExists{\@logofile}{\includegraphics[width=10mm]{\@logofile}}{}
  }
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0.3pt}
  \renewcommand{\footrulewidth}{0pt}
}

   
\newcommand{\coverpage}{%
  \thispagestyle{empty}
  \begin{center}
    \MakeUppercase{\@upperuniname}\\[4pt]
    \MakeUppercase{\@uniname}\\[4pt]
    \MakeUppercase{\@deptname}

    \vspace*{2cm}

    \IfFileExists{\@logofile}{\includegraphics[width=3.2cm]{\@logofile}}{}

    \vspace*{1.8cm}
    {\bfseries\fontsize{16}{18}\selectfont \@coursename\par}

    \vspace{0.8cm}
    {\bfseries\fontsize{16}{18}\selectfont \@reporttype\par}

    \vspace{1.0cm}
    {\bfseries\fontsize{20}{24}\selectfont \@title\par}

    \vfill
    \MakeUppercase{\@reportplace, \@reporttime}
  \end{center}
  \clearpage
}

\newcommand{\coversheet}{%
  \thispagestyle{empty}
  \begin{center}
    {\bfseries\fontsize{16}{18}\selectfont COVER SHEET}
  \end{center}

  \vspace{0.6cm}
  \noindent\textbf{Course:} \@coursename

  \noindent\textbf{Project title:} \@title

  \noindent\textbf{Supervising lecturer:} \@advisor

  \noindent\textbf{Group:} \@groupname

  \vspace{0.5cm}
  \begin{center}
    \begin{longtable}{|c|p{4cm}|c|p{3cm}|p{2.3cm}|c|}
      \hline
      \textbf{No.} & \textbf{Full name} & \textbf{Student ID} & \textbf{Major} & \textbf{Role} & \textbf{Contribution} \\
      \hline
      \endfirsthead
      \hline
      \textbf{No.} & \textbf{Full name} & \textbf{Student ID} & \textbf{Major} & \textbf{Role} & \textbf{Contribution} \\
      \hline
      \endhead
      \@membertable
      \hline
    \end{longtable}
  \end{center}

  \clearpage
}
\addto\captionsenglish{%
  \renewcommand{\refname}{References}
}
\AtBeginDocument{%
  \renewcommand{\refname}{References}
}
\makeatother
\endinput
