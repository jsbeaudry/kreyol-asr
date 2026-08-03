/**
 * Ready-made Kreyòl text so a visitor can hear the voices without writing
 * anything. Topics are one to three sentences; the paragraphs are long enough
 * to exercise segmentation and joining.
 */

export type Sample = { label: string; text: string };

export const TOPIC_SAMPLES: Sample[] = [
  {
    label: "Fanmi",
    text:
      "Fanmi se youn nan pi gwo richès yon moun ka genyen. Lè nou pran swen youn lòt, " +
      "nou vin pi fò ansanm. Menm lè lavi difisil, yon fanmi ki ini toujou jwenn yon " +
      "fason pou avanse. Lanmou, respè, ak solidarite se baz tout bon relasyon.",
  },
  {
    label: "Antreprenarya",
    text:
      "Anpil jèn ann Ayiti ap chèche kreye pwòp biznis yo olye yo tann yon djòb. Avèk " +
      "teknoloji ak entènèt, yo kapab vann pwodwi, ofri sèvis, epi jwenn kliyan menm " +
      "deyò peyi a. Chak ti pa yo fè se yon envestisman nan lavni yo.",
  },
  {
    label: "Edikasyon",
    text:
      "Edikasyon se kle ki louvri anpil pòt. Lè yon timoun aprann li, ekri, epi " +
      "reflechi, li gen plis chans pou l reyisi. Chak pwofesè, chak paran, ak chak " +
      "elèv gen yon wòl enpòtan pou konstwi yon sosyete ki pi fò.",
  },
  {
    label: "Agrikilti",
    text:
      "Agrikilti toujou rete youn nan poto mitan ekonomi peyi a. Lè kiltivatè yo jwenn " +
      "bon zouti, bon grenn, ak sipò finansye, yo kapab pwodui plis manje epi amelyore " +
      "lavi fanmi yo. Konsome pwodwi lokal ede tout kominote a grandi.",
  },
  {
    label: "Teknoloji",
    text:
      "Teknoloji ap chanje fason nou viv chak jou. Avèk yon telefòn entelijan, yon moun " +
      "ka aprann yon nouvo metye, voye lajan, pale ak fanmi l, oswa menm kòmanse yon " +
      "biznis. Pi plis moun gen aksè ak teknoloji, se pi plis opòtinite ki louvri.",
  },
  {
    label: "Enklizyon finansye",
    text:
      "Gen anpil moun ki poko gen aksè ak sèvis finansye modèn. Lè yo kapab ekonomize " +
      "lajan, resevwa peman dijital, oswa fè transfè rapid, sa rann lavi yo pi fasil. " +
      "Enklizyon finansye bay plis sekirite ak plis opòtinite pou tout moun.",
  },
  {
    label: "Kominote",
    text:
      "Chak kominote devlope lè moun yo travay ansanm. Lè vwazen yo ede youn lòt, " +
      "òganize aktivite, epi pran swen espas yo, tout moun benefisye. Yon ti aksyon " +
      "pozitif kapab chanje lavi anpil moun.",
  },
  {
    label: "Sante",
    text:
      "Sante se premye richès yon moun. Li enpòtan pou manje byen, bwè dlo pwòp, fè " +
      "egzèsis regilyèman, epi pran repo. Lè nou pran swen kò nou ak lespri nou, nou " +
      "gen plis enèji pou pouswiv rèv nou.",
  },
  {
    label: "Anviwònman",
    text:
      "Pwoteje anviwònman an se yon responsablite nou tout. Lè nou plante pye bwa, " +
      "evite jete fatra nenpòt kote, epi itilize resous yo avèk sajès, nou prepare yon " +
      "pi bon avni pou jenerasyon kap vini yo.",
  },
  {
    label: "Espwa",
    text:
      "Malgre tout defi yo, espwa toujou egziste. Chak nouvo maten pote yon lòt chans " +
      "pou aprann, travay, epi amelyore lavi nou. Avèk detèminasyon, disiplin, ak " +
      "solidarite, nou kapab bati yon avni ki pi bon pou tout moun.",
  },
];

export const PARAGRAPH_SAMPLES: Sample[] = [
  {
    label: "Paragraf 1",
    text:
      "5 Desanm se yon jou enpòtan pou tout sitwayen. Se jou kote chak moun gen " +
      "opòtinite pou fè tande vwa li atravè yon aksyon senp men ki gen anpil valè: " +
      "vote. Chak vòt konte, kèlkeswa laj, metye, oswa kote w ap viv. Lè nou patisipe " +
      "nan eleksyon, nou montre nou enterese nan avni peyi a epi nou pran responsablite " +
      "nou kòm sitwayen. Pa kite lòt moun deside pou ou. Pran kèk minit nan jounen an, " +
      "ale nan sant vòt ou, epi egzèse dwa ou avèk fyète ak respè.",
  },
  {
    label: "Paragraf 2",
    text:
      "Lavni yon peyi pa bati sèlman ak pawòl; li bati ak aksyon chak sitwayen. Vote se " +
      "youn nan pi gwo fason pou patisipe nan lavi demokratik peyi a. Lè ou vote, ou " +
      "ede chwazi direksyon peyi a pral pran pou plizyè ane. Kit se premye fwa w ap " +
      "vote oswa ou deja gen eksperyans, patisipasyon w toujou enpòtan. Ankouraje " +
      "fanmi w, zanmi w, ak vwazen w pou yo leve bonè, verifye kote y ap vote, epi ale " +
      "fè devwa sivik yo. Ansanm, nou ka montre fòs yon pèp ki angaje.",
  },
  {
    label: "Paragraf 3",
    text:
      "Demokrasi vin pi solid lè plis sitwayen patisipe. Se poutèt sa 5 Desanm pa ta " +
      "dwe yon jou tankou nenpòt lòt. Se yon jou pou mete enterè peyi a an premye, pou " +
      "montre nou kwè nan pouvwa patisipasyon, epi pou fè tande opinyon nou nan yon " +
      "fason pasifik. Kèlkeswa chwa ou, sa ki pi enpòtan an se ale vote. Pa kite move " +
      "tan, distans, oswa dekourajman anpeche w egzèse dwa ou. Chak vòt se yon etap pou " +
      "konstwi yon pi bon avni.",
  },
  {
    label: "Paragraf 4",
    text:
      "Chak jenerasyon gen yon responsablite anvè jenerasyon k ap vini yo. Desizyon nou " +
      "pran jodi a kapab gen enpak pandan plizyè ane. Lè ou vote 5 Desanm, ou pa sèlman " +
      "pran yon desizyon pou tèt ou; ou patisipe nan desizyon ki kapab enfliyanse lavi " +
      "fanmi w, kominote w, ak tout peyi a. Fè vwa w konte. Ale vote avèk kalm, avèk " +
      "respè, epi ankouraje lòt moun fè menm jan an. Yon gwo patisipasyon se yon siy " +
      "angajman sitwayen yo anvè demokrasi.",
  },
  {
    label: "Paragraf 5",
    text:
      "Si chak moun fè pati pa yo, ansanm nou kapab fè yon gwo diferans. Jou eleksyon " +
      "an se yon okazyon pou montre lanmou nou pou peyi a atravè yon aksyon konkrè. " +
      "Prepare davans, verifye dokiman ou bezwen yo si sa nesesè, epi planifye lè w ap " +
      "ale vote. Pale ak fanmi w ak zanmi w pou raple yo enpòtans patisipasyon an. " +
      "5 Desanm, pa rete lakay ou. Soti, ale vote, epi fè vwa w konte. Chak vòt gen " +
      "valè, e chak sitwayen gen yon wòl enpòtan nan bati avni peyi a.",
  },
];
