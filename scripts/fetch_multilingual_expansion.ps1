param(
    [string]$TrainPath = 'data/raw/multilingual_expansion_train.txt',
    [string]$ValidationPath = 'data/raw/multilingual_expansion_validation.txt',
    [string]$ManifestPath = 'data/raw/multilingual_expansion_sources.json',
    [int]$RequestDelaySeconds = 30,
    [string]$Contact = ''
)
$ErrorActionPreference = 'Stop'
if ($RequestDelaySeconds -lt 30) { throw 'RequestDelaySeconds must be at least 30 seconds for Wikimedia requests.' }

# Eight training leads and two disjoint validation leads per language.
$pages = @(
    @{ lang='en'; title='Artificial intelligence'; split='train' },
    @{ lang='en'; title='Computer science'; split='train' },
    @{ lang='en'; title='Deep learning'; split='train' },
    @{ lang='en'; title='Artificial neural network'; split='train' },
    @{ lang='en'; title='Computational linguistics'; split='train' },
    @{ lang='en'; title='Speech recognition'; split='train' },
    @{ lang='en'; title='Computer vision'; split='train' },
    @{ lang='en'; title='Information retrieval'; split='train' },
    @{ lang='en'; title='Data mining'; split='validation' },
    @{ lang='en'; title='Recommender system'; split='validation' },

    @{ lang='es'; title='Inteligencia artificial'; split='train' },
    @{ lang='es'; title='Ciencias de la computación'; split='train' },
    @{ lang='es'; title='Aprendizaje profundo'; split='train' },
    @{ lang='es'; title='Red neuronal artificial'; split='train' },
    @{ lang='es'; title='Lingüística computacional'; split='train' },
    @{ lang='es'; title='Reconocimiento del habla'; split='train' },
    @{ lang='es'; title='Visión artificial'; split='train' },
    @{ lang='es'; title='Recuperación de información'; split='train' },
    @{ lang='es'; title='Minería de datos'; split='validation' },
    @{ lang='es'; title='Sistema de recomendación'; split='validation' },

    @{ lang='fr'; title='Intelligence artificielle'; split='train' },
    @{ lang='fr'; title='Informatique'; split='train' },
    @{ lang='fr'; title='Apprentissage profond'; split='train' },
    @{ lang='fr'; title='Réseau de neurones artificiels'; split='train' },
    @{ lang='fr'; title='Linguistique informatique'; split='train' },
    @{ lang='fr'; title='Reconnaissance automatique de la parole'; split='train' },
    @{ lang='fr'; title='Vision par ordinateur'; split='train' },
    @{ lang='fr'; title="Recherche d'information"; split='train' },
    @{ lang='fr'; title='Exploration de données'; split='validation' },
    @{ lang='fr'; title='Système de recommandation'; split='validation' },

    @{ lang='ja'; title='人工知能'; split='train' },
    @{ lang='ja'; title='計算機科学'; split='train' },
    @{ lang='ja'; title='深層学習'; split='train' },
    @{ lang='ja'; title='人工ニューラルネットワーク'; split='train' },
    @{ lang='ja'; title='計算言語学'; split='train' },
    @{ lang='ja'; title='音声認識'; split='train' },
    @{ lang='ja'; title='コンピュータビジョン'; split='train' },
    @{ lang='ja'; title='情報検索'; split='train' },
    @{ lang='ja'; title='データマイニング'; split='validation' },
    @{ lang='ja'; title='推薦システム'; split='validation' },

    @{ lang='zh'; title='人工智能'; split='train' },
    @{ lang='zh'; title='计算机科学'; split='train' },
    @{ lang='zh'; title='深度学习'; split='train' },
    @{ lang='zh'; title='人工神经网络'; split='train' },
    @{ lang='zh'; title='计算语言学'; split='train' },
    @{ lang='zh'; title='语音识别'; split='train' },
    @{ lang='zh'; title='计算机视觉'; split='train' },
    @{ lang='zh'; title='信息检索'; split='train' },
    @{ lang='zh'; title='数据挖掘'; split='validation' },
    @{ lang='zh'; title='推薦系統'; split='validation' },

    @{ lang='de'; title='Künstliche Intelligenz'; split='train' },
    @{ lang='de'; title='Informatik'; split='train' },
    @{ lang='de'; title='Deep Learning'; split='train' },
    @{ lang='de'; title='Künstliches neuronales Netz'; split='train' },
    @{ lang='de'; title='Computerlinguistik'; split='train' },
    @{ lang='de'; title='Spracherkennung'; split='train' },
    @{ lang='de'; title='Computervision'; split='train' },
    @{ lang='de'; title='Information Retrieval'; split='train' },
    @{ lang='de'; title='Data-Mining'; split='validation' },
    @{ lang='de'; title='Empfehlungssystem'; split='validation' },

    @{ lang='pt'; title='Inteligência artificial'; split='train' },
    @{ lang='pt'; title='Ciência da computação'; split='train' },
    @{ lang='pt'; title='Aprendizado de máquina'; split='train' },
    @{ lang='pt'; title='Aprendizagem profunda'; split='train' },
    @{ lang='pt'; title='Rede neural artificial'; split='train' },
    @{ lang='pt'; title='Reconhecimento de fala'; split='train' },
    @{ lang='pt'; title='Visão computacional'; split='train' },
    @{ lang='pt'; title='Recuperação de informação'; split='train' },
    @{ lang='pt'; title='Mineração de dados'; split='validation' },
    @{ lang='pt'; title='Sistema de recomendação'; split='validation' },

    @{ lang='ru'; title='Искусственный интеллект'; split='train' },
    @{ lang='ru'; title='Информатика'; split='train' },
    @{ lang='ru'; title='Глубокое обучение'; split='train' },
    @{ lang='ru'; title='Искусственная нейронная сеть'; split='train' },
    @{ lang='ru'; title='Компьютерная лингвистика'; split='train' },
    @{ lang='ru'; title='Распознавание речи'; split='train' },
    @{ lang='ru'; title='Компьютерное зрение'; split='train' },
    @{ lang='ru'; title='Информационный поиск'; split='train' },
    @{ lang='ru'; title='Интеллектуальный анализ данных'; split='validation' },
    @{ lang='ru'; title='Рекомендательная система'; split='validation' },

    @{ lang='ar'; title='ذكاء اصطناعي'; split='train' },
    @{ lang='ar'; title='علم الحاسوب'; split='train' },
    @{ lang='ar'; title='تعلم عميق'; split='train' },
    @{ lang='ar'; title='شبكة عصبونية'; split='train' },
    @{ lang='ar'; title='اللسانيات الحاسوبية'; split='train' },
    @{ lang='ar'; title='التعرف على الكلام'; split='train' },
    @{ lang='ar'; title='رؤية حاسوبية'; split='train' },
    @{ lang='ar'; title='استرجاع المعلومات'; split='train' },
    @{ lang='ar'; title='تنقيب البيانات'; split='validation' },
    @{ lang='ar'; title='نظام التوصية'; split='validation' },

    @{ lang='en'; title='Natural language processing'; split='train' },
    @{ lang='en'; title='Data science'; split='train' },
    @{ lang='en'; title='Robotics'; split='train' },
    @{ lang='en'; title='Probability theory'; split='train' },
    @{ lang='en'; title='Statistics'; split='train' },
    @{ lang='en'; title='Computer network'; split='train' },
    @{ lang='en'; title='Software engineering'; split='train' },
    @{ lang='en'; title='Information theory'; split='train' },
    @{ lang='en'; title='Database'; split='validation' },
    @{ lang='en'; title='Mathematics'; split='validation' },

    @{ lang='es'; title='Procesamiento de lenguaje natural'; split='train' },
    @{ lang='es'; title='Ciencia de datos'; split='train' },
    @{ lang='es'; title='Robótica'; split='train' },
    @{ lang='es'; title='Teoría de la probabilidad'; split='train' },
    @{ lang='es'; title='Estadística'; split='train' },
    @{ lang='es'; title='Red de computadoras'; split='train' },
    @{ lang='es'; title='Ingeniería de software'; split='train' },
    @{ lang='es'; title='Teoría de la información'; split='train' },
    @{ lang='es'; title='Base de datos'; split='validation' },
    @{ lang='es'; title='Matemáticas'; split='validation' },

    @{ lang='fr'; title='Traitement automatique des langues'; split='train' },
    @{ lang='fr'; title='Science des données'; split='train' },
    @{ lang='fr'; title='Robotique'; split='train' },
    @{ lang='fr'; title='Théorie des probabilités'; split='train' },
    @{ lang='fr'; title='Statistique'; split='train' },
    @{ lang='fr'; title='Réseau informatique'; split='train' },
    @{ lang='fr'; title='Génie logiciel'; split='train' },
    @{ lang='fr'; title="Théorie de l'information"; split='train' },
    @{ lang='fr'; title='Base de données'; split='validation' },
    @{ lang='fr'; title='Mathématiques'; split='validation' },

    @{ lang='ja'; title='自然言語処理'; split='train' },
    @{ lang='ja'; title='データサイエンス'; split='train' },
    @{ lang='ja'; title='ロボット工学'; split='train' },
    @{ lang='ja'; title='確率論'; split='train' },
    @{ lang='ja'; title='統計学'; split='train' },
    @{ lang='ja'; title='コンピュータネットワーク'; split='train' },
    @{ lang='ja'; title='ソフトウェア工学'; split='train' },
    @{ lang='ja'; title='情報理論'; split='train' },
    @{ lang='ja'; title='データベース'; split='validation' },
    @{ lang='ja'; title='数学'; split='validation' },

    @{ lang='zh'; title='自然语言处理'; split='train' },
    @{ lang='zh'; title='数据科学'; split='train' },
    @{ lang='zh'; title='机器人学'; split='train' },
    @{ lang='zh'; title='概率论'; split='train' },
    @{ lang='zh'; title='统计学'; split='train' },
    @{ lang='zh'; title='计算机网络'; split='train' },
    @{ lang='zh'; title='软件工程'; split='train' },
    @{ lang='zh'; title='信息论'; split='train' },
    @{ lang='zh'; title='数据库'; split='validation' },
    @{ lang='zh'; title='数学'; split='validation' },

    @{ lang='de'; title='Datenkompression'; split='train' },
    @{ lang='de'; title='Data Science'; split='train' },
    @{ lang='de'; title='Robotik'; split='train' },
    @{ lang='de'; title='Wahrscheinlichkeitstheorie'; split='train' },
    @{ lang='de'; title='Statistik'; split='train' },
    @{ lang='de'; title='Rechnernetz'; split='train' },
    @{ lang='de'; title='Softwaretechnik'; split='train' },
    @{ lang='de'; title='Informationstheorie'; split='train' },
    @{ lang='de'; title='Datenbank'; split='validation' },
    @{ lang='de'; title='Mathematik'; split='validation' },

    @{ lang='pt'; title='Processamento de linguagem natural'; split='train' },
    @{ lang='pt'; title='Ciência de dados'; split='train' },
    @{ lang='pt'; title='Robótica'; split='train' },
    @{ lang='pt'; title='Teoria das probabilidades'; split='train' },
    @{ lang='pt'; title='Estatística'; split='train' },
    @{ lang='pt'; title='Rede de computadores'; split='train' },
    @{ lang='pt'; title='Engenharia de software'; split='train' },
    @{ lang='pt'; title='Teoria da informação'; split='train' },
    @{ lang='pt'; title='Base de dados'; split='validation' },
    @{ lang='pt'; title='Matemática'; split='validation' },

    @{ lang='ru'; title='Обработка естественного языка'; split='train' },
    @{ lang='ru'; title='Наука о данных'; split='train' },
    @{ lang='ru'; title='Робототехника'; split='train' },
    @{ lang='ru'; title='Теория вероятностей'; split='train' },
    @{ lang='ru'; title='Статистика'; split='train' },
    @{ lang='ru'; title='Компьютерная сеть'; split='train' },
    @{ lang='ru'; title='Программная инженерия'; split='train' },
    @{ lang='ru'; title='Теория информации'; split='train' },
    @{ lang='ru'; title='База данных'; split='validation' },
    @{ lang='ru'; title='Математика'; split='validation' },

    @{ lang='ar'; title='معالجة اللغات الطبيعية'; split='train' },
    @{ lang='ar'; title='علم البيانات'; split='train' },
    @{ lang='ar'; title='روبوت'; split='train' },
    @{ lang='ar'; title='نظرية الاحتمالات'; split='train' },
    @{ lang='ar'; title='إحصاء'; split='train' },
    @{ lang='ar'; title='شبكة الحاسوب'; split='train' },
    @{ lang='ar'; title='هندسة البرمجيات'; split='train' },
    @{ lang='ar'; title='نظرية المعلومات'; split='train' },
    @{ lang='ar'; title='قاعدة بيانات'; split='validation' },
    @{ lang='ar'; title='رياضيات'; split='validation' },

    # Additional language batches use eight train leads and two disjoint held-out leads.
    @{ lang='hi'; title='कृत्रिम बुद्धिमत्ता'; split='train' },
    @{ lang='hi'; title='कंप्यूटर विज्ञान'; split='train' },
    @{ lang='hi'; title='मशीन अधिगम'; split='train' },
    @{ lang='hi'; title='गहन शिक्षण'; split='train' },
    @{ lang='hi'; title='कृत्रिम तंत्रिका नेटवर्क'; split='train' },
    @{ lang='hi'; title='प्राकृतिक भाषा संसाधन'; split='train' },
    @{ lang='hi'; title='वाक् पहचान'; split='train' },
    @{ lang='hi'; title='सॉफ्टवेयर इंजीनियरिंग'; split='train' },
    @{ lang='hi'; title='सांख्यिकी'; split='validation' },
    @{ lang='hi'; title='रोबोटिक्स'; split='validation' },

    @{ lang='bn'; title='কৃত্রিম বুদ্ধিমত্তা'; split='train' },
    @{ lang='bn'; title='কম্পিউটার বিজ্ঞান'; split='train' },
    @{ lang='bn'; title='মেশিন লার্নিং'; split='train' },
    @{ lang='bn'; title='গভীর শিক্ষা'; split='train' },
    @{ lang='bn'; title='কৃত্রিম নিউরাল নেটওয়ার্ক'; split='train' },
    @{ lang='bn'; title='প্রাকৃতিক ভাষা প্রক্রিয়াকরণ'; split='train' },
    @{ lang='bn'; title='বাক্‌ শনাক্তকরণ'; split='train' },
    @{ lang='bn'; title='সফটওয়্যার প্রকৌশল'; split='train' },
    @{ lang='bn'; title='পরিসংখ্যান'; split='validation' },
    @{ lang='bn'; title='রোবোটিক্স'; split='validation' },

    @{ lang='id'; title='kecerdasan buatan'; split='train' },
    @{ lang='id'; title='ilmu komputer'; split='train' },
    @{ lang='id'; title='pembelajaran mesin'; split='train' },
    @{ lang='id'; title='pembelajaran mendalam'; split='train' },
    @{ lang='id'; title='jaringan saraf tiruan'; split='train' },
    @{ lang='id'; title='pemrosesan bahasa alami'; split='train' },
    @{ lang='id'; title='pengenalan ucapan'; split='train' },
    @{ lang='id'; title='rekayasa perangkat lunak'; split='train' },
    @{ lang='id'; title='statistika'; split='validation' },
    @{ lang='id'; title='robotika'; split='validation' },

    @{ lang='tr'; title='yapay zekâ'; split='train' },
    @{ lang='tr'; title='bilgisayar bilimi'; split='train' },
    @{ lang='tr'; title='makine öğrenimi'; split='train' },
    @{ lang='tr'; title='derin öğrenme'; split='train' },
    @{ lang='tr'; title='yapay sinir ağı'; split='train' },
    @{ lang='tr'; title='doğal dil işleme'; split='train' },
    @{ lang='tr'; title='konuşma tanıma'; split='train' },
    @{ lang='tr'; title='yazılım mühendisliği'; split='train' },
    @{ lang='tr'; title='istatistik'; split='validation' },
    @{ lang='tr'; title='robotik'; split='validation' },

    @{ lang='vi'; title='trí tuệ nhân tạo'; split='train' },
    @{ lang='vi'; title='khoa học máy tính'; split='train' },
    @{ lang='vi'; title='học máy'; split='train' },
    @{ lang='vi'; title='học sâu'; split='train' },
    @{ lang='vi'; title='mạng nơ-ron nhân tạo'; split='train' },
    @{ lang='vi'; title='xử lý ngôn ngữ tự nhiên'; split='train' },
    @{ lang='vi'; title='nhận dạng tiếng nói'; split='train' },
    @{ lang='vi'; title='kỹ nghệ phần mềm'; split='train' },
    @{ lang='vi'; title='thống kê'; split='validation' },
    @{ lang='vi'; title='robot'; split='validation' },

    @{ lang='ko'; title='인공지능'; split='train' },
    @{ lang='ko'; title='컴퓨터 과학'; split='train' },
    @{ lang='ko'; title='기계 학습'; split='train' },
    @{ lang='ko'; title='딥 러닝'; split='train' },
    @{ lang='ko'; title='인공 신경망'; split='train' },
    @{ lang='ko'; title='자연어 처리'; split='train' },
    @{ lang='ko'; title='음성 인식'; split='train' },
    @{ lang='ko'; title='소프트웨어 공학'; split='train' },
    @{ lang='ko'; title='통계학'; split='validation' },
    @{ lang='ko'; title='로봇공학'; split='validation' },

    @{ lang='it'; title='intelligenza artificiale'; split='train' },
    @{ lang='it'; title='informatica'; split='train' },
    @{ lang='it'; title='apprendimento automatico'; split='train' },
    @{ lang='it'; title='apprendimento profondo'; split='train' },
    @{ lang='it'; title='rete neurale artificiale'; split='train' },
    @{ lang='it'; title='elaborazione del linguaggio naturale'; split='train' },
    @{ lang='it'; title='riconoscimento vocale'; split='train' },
    @{ lang='it'; title='ingegneria del software'; split='train' },
    @{ lang='it'; title='statistica'; split='validation' },
    @{ lang='it'; title='robotica'; split='validation' },

    @{ lang='nl'; title='kunstmatige intelligentie'; split='train' },
    @{ lang='nl'; title='informatica'; split='train' },
    @{ lang='nl'; title='machinaal leren'; split='train' },
    @{ lang='nl'; title='diep leren'; split='train' },
    @{ lang='nl'; title='kunstmatig neuraal netwerk'; split='train' },
    @{ lang='nl'; title='computationele taalkunde'; split='train' },
    @{ lang='nl'; title='spraakherkenning'; split='train' },
    @{ lang='nl'; title='software engineering'; split='train' },
    @{ lang='nl'; title='statistiek'; split='validation' },
    @{ lang='nl'; title='robotica'; split='validation' },

    @{ lang='pl'; title='sztuczna inteligencja'; split='train' },
    @{ lang='pl'; title='informatyka'; split='train' },
    @{ lang='pl'; title='uczenie maszynowe'; split='train' },
    @{ lang='pl'; title='uczenie głębokie'; split='train' },
    @{ lang='pl'; title='sztuczna sieć neuronowa'; split='train' },
    @{ lang='pl'; title='przetwarzanie języka naturalnego'; split='train' },
    @{ lang='pl'; title='rozpoznawanie mowy'; split='train' },
    @{ lang='pl'; title='inżynieria oprogramowania'; split='train' },
    @{ lang='pl'; title='statystyka'; split='validation' },
    @{ lang='pl'; title='robotyka'; split='validation' },

    @{ lang='fa'; title='هوش مصنوعی'; split='train' },
    @{ lang='fa'; title='علوم رایانه'; split='train' },
    @{ lang='fa'; title='یادگیری ماشین'; split='train' },
    @{ lang='fa'; title='یادگیری عمیق'; split='train' },
    @{ lang='fa'; title='شبکه عصبی مصنوعی'; split='train' },
    @{ lang='fa'; title='پردازش زبان طبیعی'; split='train' },
    @{ lang='fa'; title='تشخیص گفتار'; split='train' },
    @{ lang='fa'; title='مهندسی نرم‌افزار'; split='train' },
    @{ lang='fa'; title='آمار'; split='validation' },
    @{ lang='fa'; title='رباتیک'; split='validation' },

    @{ lang='sv'; title='artificiell intelligens'; split='train' },
    @{ lang='sv'; title='datavetenskap'; split='train' },
    @{ lang='sv'; title='maskininlärning'; split='train' },
    @{ lang='sv'; title='djupinlärning'; split='train' },
    @{ lang='sv'; title='artificiellt neuronnät'; split='train' },
    @{ lang='sv'; title='datorlingvistik'; split='train' },
    @{ lang='sv'; title='taligenkänning'; split='train' },
    @{ lang='sv'; title='programvaruteknik'; split='train' },
    @{ lang='sv'; title='statistik'; split='validation' },
    @{ lang='sv'; title='robotik'; split='validation' }
)

$records = [System.Collections.Generic.List[object]]::new()
$missingTitles = [System.Collections.Generic.List[object]]::new()
$trainText = [System.Collections.Generic.List[string]]::new()
$validationText = [System.Collections.Generic.List[string]]::new()
$cacheRoot = 'data/cache/wikipedia_expansion'
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
$contactPart = if ($Contact) { "; contact: $Contact" } else { '' }
$userAgent = "AMA-10 multilingual corpus expansion fetcher/1.0 (local research pipeline)$contactPart"

foreach ($group in ($pages | Group-Object lang)) {
    $language = $group.Name
    $cachePath = Join-Path $cacheRoot "$language.json"
    $cachedByTitle = @{}
    $groupCache = [System.Collections.Generic.List[object]]::new()
    $cacheNeedsRewrite = $false
    if (Test-Path -LiteralPath $cachePath) {
        $cached = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $cachePath),[Text.UTF8Encoding]::new($false)) | ConvertFrom-Json
        $cachedRecords = @($cached)
        foreach ($item in $cachedRecords) {
            $cachedTitle = if ($item.metadata.requested_title) { $item.metadata.requested_title } else { $item.metadata.title }
            $cachedByTitle[$cachedTitle.Replace('_',' ').ToLowerInvariant()] = $item
        }
    }
    $missingSpecs = [System.Collections.Generic.List[object]]::new()
    foreach ($spec in $group.Group) {
        $key = $spec.title.Replace('_',' ').ToLowerInvariant()
        if ($cachedByTitle.ContainsKey($key)) {
            $item = $cachedByTitle[$key]
            if ($item.metadata.split -ne $spec.split) { throw "Cached split mismatch for $language`: $($spec.title)" }
            if (-not $item.metadata.PSObject.Properties['requested_title']) {
                $item.metadata | Add-Member -MemberType NoteProperty -Name requested_title -Value $spec.title
                $cacheNeedsRewrite = $true
            }
            $records.Add($item.metadata)
            $document = "$($item.metadata.title)`n$($item.extract)`n"
            if ($spec.split -eq 'train') { $trainText.Add($document) }
            else { $validationText.Add($document) }
            $groupCache.Add($item)
        } else { $missingSpecs.Add($spec) }
    }
    if ($missingSpecs.Count -eq 0) {
        if ($cacheNeedsRewrite) { $groupCache | ConvertTo-Json -Depth 7 | Set-Content -Encoding utf8 $cachePath }
        Write-Host "Using cached successful $language page batch."
        continue
    }
    $titles = @($missingSpecs | ForEach-Object { $_.title })
    $joinedTitles = [Uri]::EscapeDataString([string]::Join('|', $titles))
    $uri = "https://$language.wikipedia.org/w/api.php?action=query&format=json&redirects=1&prop=extracts%7Cinfo%7Crevisions&explaintext=1&exintro=1&exlimit=max&inprop=url&rvprop=ids&maxlag=5&titles=$joinedTitles"
    $response = $null
    for ($attempt = 0; $attempt -lt 4 -and -not $response; $attempt++) {
        $rateLimited = $false
        try {
            $response = Invoke-RestMethod -Uri $uri -UserAgent $userAgent -TimeoutSec 60
            if ($response.error) {
                $rateLimited = $response.error.code -match 'ratelimit|too-many-requests'
                $message = $response.error.info
                $response = $null
                throw $message
            }
        } catch {
            if ($attempt -eq 3) { throw "Wikipedia fetch failed for $language`: $_" }
            $response = $null
            $waitSeconds = [int][Math]::Min(600, 60 * [Math]::Pow(2, $attempt))
            $statusCode = 0
            $retryAfter = $null
            try {
                $statusCode = [int]$_.Exception.Response.StatusCode
                $retryAfter = $_.Exception.Response.Headers['Retry-After']
            } catch { }
            if ($rateLimited -or $statusCode -eq 429) {
                $waitSeconds = [Math]::Max(120, $waitSeconds)
            }
            $retrySeconds = 0
            if ([int]::TryParse([string]$retryAfter,[ref]$retrySeconds)) {
                $waitSeconds = [Math]::Max($waitSeconds,[Math]::Min(600,$retrySeconds))
            }
            Write-Warning "Request throttled or failed for $language; waiting $waitSeconds seconds before retry."
            Start-Sleep -Seconds $waitSeconds
        }
    }
    $byTitle = @{}
    foreach ($page in $response.query.pages.PSObject.Properties.Value) {
        if (-not $page.missing -and -not [string]::IsNullOrWhiteSpace($page.extract)) {
            $byTitle[$page.title.Replace('_',' ').ToLowerInvariant()] = $page
        }
    }
    $aliases = @{}
    foreach ($mapping in @($response.query.normalized) + @($response.query.redirects) + @($response.query.converted)) {
        if ($mapping.from -and $mapping.to) {
            $aliases[$mapping.from.Replace('_',' ').ToLowerInvariant()] = $mapping.to.Replace('_',' ').ToLowerInvariant()
        }
    }
    foreach ($spec in $missingSpecs) {
        $key = $spec.title.Replace('_',' ').ToLowerInvariant()
        if ($aliases.ContainsKey($key)) { $key = $aliases[$key] }
        $page = $byTitle[$key]
        if (-not $page) {
            $missingTitles.Add([pscustomobject]@{ language=$language; requested_title=$spec.title; split=$spec.split; reason='No non-empty lead extract in response' })
            Write-Warning "No extract found for $language`: $($spec.title); keeping the remaining successful pages."
            continue
        }
        $document = "$($page.title)`n$($page.extract)`n"
        if ($spec.split -eq 'train') { $trainText.Add($document) }
        else { $validationText.Add($document) }
        $metadata = [pscustomobject]@{
            language = $language
            requested_title = $spec.title
            title = $page.title
            pageid = $page.pageid
            revision = $page.revisions[0].revid
            url = $page.fullurl
            api_url = $uri
            extract_scope = 'lead introduction'
            retrieval_utc = (Get-Date).ToUniversalTime().ToString('o')
            license = 'CC BY-SA 4.0'
            license_url = 'https://creativecommons.org/licenses/by-sa/4.0/'
            split = $spec.split
        }
        $records.Add($metadata)
        $groupCache.Add([pscustomobject]@{ metadata=$metadata; extract=$page.extract })
        Write-Host "$language`: $($page.title) chars=$($page.extract.Length) split=$($spec.split)"
    }
    $groupCache | ConvertTo-Json -Depth 7 | Set-Content -Encoding utf8 $cachePath
    Start-Sleep -Seconds $RequestDelaySeconds
}

foreach ($languageRecords in ($records | Group-Object language)) {
    $pageIds = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($record in $languageRecords.Group) {
        if (-not $record.pageid -or -not $pageIds.Add([string]$record.pageid)) {
            throw "Duplicate or missing page ID in $($languageRecords.Name) source records: $($record.pageid)"
        }
    }
    $trainCount = @($languageRecords.Group | Where-Object split -eq 'train').Count
    $validationCount = @($languageRecords.Group | Where-Object split -eq 'validation').Count
    if (-not $trainCount -or -not $validationCount) { throw "Both train and validation pages are required for $($languageRecords.Name)." }
}

foreach ($path in @($TrainPath,$ValidationPath,$ManifestPath)) {
    $parent = Split-Path -Parent $path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($TrainPath,($trainText -join "`n"),$utf8)
[IO.File]::WriteAllText($ValidationPath,($validationText -join "`n"),$utf8)
$records | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $ManifestPath
$missingPath = [IO.Path]::ChangeExtension($ManifestPath,'.missing.json')
ConvertTo-Json -InputObject @($missingTitles) -Depth 4 | Set-Content -Encoding utf8 $missingPath
Write-Host "Saved $($trainText.Count) training and $($validationText.Count) validation pages; skipped $($missingTitles.Count) missing titles."
