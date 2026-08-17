{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;\f1\fmodern\fcharset0 Courier-Bold;\f2\fmodern\fcharset0 Courier-Oblique;
\f3\fmodern\fcharset0 Courier-BoldOblique;\f4\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red181\green90\blue255;\red33\green189\blue138;\red52\green144\blue255;
}
{\*\expandedcolortbl;;\cssrgb\c76863\c47059\c100000;\cssrgb\c10196\c77647\c61176;\cssrgb\c24706\c64314\c100000;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # Dehydrated Tweet Dataset\cf3 \strokec3 \
\
This repository contains the 
\f1\b **Tweet identifiers**
\f0\b0  and 
\f1\b **our derived annotations**
\f0\b0 \
for the four classification tasks used in this study: hate speech, sexism, lexical bias, and sentiment.\
\
In line with the platform terms, we cannot release tweet texts or user information, but a copy of all the tweets we used can be found in the Internet Archive (\{https://archive.org/details/twitterstream). Is archive used to be publicly available but current conditions for access depend on the Internet Archive.\
\
\cf2 \strokec2 ## Files\cf3 \strokec3 \
\
| File | Task |\
|------|------|\
| `hate_tweet_ids.csv`         | Hate speech |\
| `sexism_tweet_ids.csv`       | Sexism |\
| `lexical_bias_tweet_ids.csv` | Lexical bias |\
| `sentiment_tweet_ids.csv`    | Sentiment |\
\
\cf2 \strokec2 ### Columns\cf3 \strokec3 \
\
| Column | Description |\
|--------|-------------|\
| `id`             | Tweet ID (string). Use this to rehydrate the tweet text. |\
| `mistral_label`  | Label assigned by the Mistral teacher model. |\
| `gemma_label`    | Label assigned by the Gemma teacher model. |\
| `claude_label`   | Label assigned by the Claude tiebreaker (present only when the two teachers disagreed; otherwise `null`). |\
| `final_label`    | The final label used in the study. |\
| `label_source`   | Provenance of `final_label`: `teacher_agree` (both teachers agreed) or the tiebreaker source (teachers disagreed \uc0\u8594  Claude decided). |\
\
\pard\pardeftab720\partightenfactor0

\f2\i \cf4 \strokec4 > 
\f3\b **Note:**
\f2\b0  `id` is stored as a 
\f3\b **string**
\f2\b0  \'97 do not import it as a numeric type,
\f0\i0 \cf3 \strokec3 \

\f2\i \cf4 \strokec4 > as Tweet IDs exceed 2^53 and will be corrupted (e.g. rounded/scientific
\f0\i0 \cf3 \strokec3 \

\f2\i \cf4 \strokec4 > notation) if parsed as integers or floats.
\f4\i0 \cf3 \strokec3 \
}